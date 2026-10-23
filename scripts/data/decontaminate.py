#!/usr/bin/env python3
"""Decontaminacao treino x avaliacao (spec fase-03 §4.5 etapa 9 / G3.4; fase-06 §9). MESMO script das Fases 03 e 06.

Remove do TREINO qualquer item que compartilhe com val/test/bench (e opcionalmente legacy-v1) algum dos sinais:
  exact       texto normalizado identico (codigo + instrucao, literais -> placeholders)
  ngram13     13-grama comum de tokens normalizados (literais -> <N>/<S>/<D>/<OID>/<U>; codigo e instrucoes sem acento)
  approx_key  chave aproximada (v1): operacao + colecao + conjunto de campos + conjunto de operadores
  minhash     near-dup: Jaccard estimado >= 0,7 (MinHash 128 permutacoes, shingles de 5 tokens normalizados)
  blind       hash (item/prompt) coincide com item selado -- SOMENTE via `blind_guard` (nunca le data/eval/blind);
              manifesto selado ausente => `blind: not_sealed` no relatorio (nao falha)

Uso:
  python scripts/data/decontaminate.py --in data/validated/v1/fim.jsonl data/validated/v1/chat.jsonl \
      [--legacy bakeoff/data/v1_benchmark.jsonl ...] [--out-ids removed.txt] [--report-dir reports/decontam]
Itens com `split == train` sao o treino; `val`/`test`/`bench` sao a avaliacao. Relatorio: reports/decontam/<dataset-hash>.json.
Uso programatico: `run(train_records, eval_records, ...) -> dict` (usado por validate_pipeline.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blind_guard as bg  # noqa: E402
import hom_guard  # noqa: E402

ROOT = HERE.parents[1]
NGRAM_N = 13
# Regra OFICIAL do treino (decisao do usuario, opcao B): um 13-grama comum so conta como contaminacao se tiver >= 2 tokens de
# CONTEUDO (campo/colecao/constante nomeada). 13-gramas puramente estruturais ($match/$group/$sort, pontuacao, placeholders, API do
# Console) nao contam: os schemas sao disjuntos por split (split_role), logo nao ha vazamento de conteudo; a regra literal (k=0)
# fica como auditoria (`analysis`). exact, approx_key, MinHash e blind seguem inalterados.
DEFAULT_NGRAM_MIN_CONTENT = 2
SHINGLE_K = 5
NUM_PERM = 128
BANDS, ROWS = 25, 5           # 25*5 <= 128; recall ~99% em J=0,7
DECONTAM_JACCARD = 0.7
DEDUP_JACCARD = 0.85
_P = (1 << 31) - 1
_NP = None


def _numpy():
    """numpy se disponivel (tambem via KAPI_TOKENIZERS_SITE / venv do bake-off); None => MinHash em Python puro (mesmo resultado)."""
    global _NP
    if _NP is None:
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            import os
            site = os.environ.get("KAPI_TOKENIZERS_SITE") or r"C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages"
            np = None
            if site and Path(site).exists() and site not in sys.path:
                sys.path.append(site)
                try:
                    import numpy as np  # noqa: F811
                except Exception:
                    np = None
        _NP = np if np is not None else False
    return _NP or None
EVAL_SPLITS = ("val", "test", "bench")
RULES = ("blind", "exact", "ngram13", "approx_key", "minhash")

# --------------------------------------------------------------------------- tokenizacao
_CODE_RE = re.compile(r"""
    (?P<ws>\s+) | (?P<lc>//[^\n]*) | (?P<bc>/\*.*?\*/) |
    (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`) |
    (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) |
    (?P<id>[A-Za-z_$][\w$]*) |
    (?P<p>.)
""", re.X | re.S)
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
_OID = re.compile(r"^[0-9a-fA-F]{24}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_NAME_FUNCS = {"getCollection", "getDatabase", "getConnection", "getSiblingDB", "use"}
_UNESC = re.compile(r"\\(.)", re.S)


def code_tokens(text: str, literals: str = "placeholder") -> list[str]:
    """Tokens do codigo. `literals='placeholder'` (decontaminacao): numeros -> <N>, strings -> <S>/<D>/<OID>/<U>, chaves
    entre aspas viram identificadores, `$campo` e nomes de colecao/banco ficam. `literals='keep'` (dedup): literais
    preservados (so layout/aspas/espacos sao normalizados). Comentarios sao descartados."""
    out: list[str] = []
    pos, n = 0, len(text)
    while pos < n:
        m = _CODE_RE.match(text, pos)
        pos = m.end()
        k = m.lastgroup
        if k in ("ws", "lc", "bc"):
            continue
        v = m.group()
        if k == "num":
            out.append("<N>" if literals == "placeholder" else "N:" + v)
        elif k == "str":
            body = _UNESC.sub(r"\1", v[1:-1])
            nxt = text[pos:pos + 40].lstrip()
            if nxt.startswith(":"):                      # chave de objeto: nome de campo/operador
                out.append(body)
            elif literals == "keep":
                out.append("S:" + body)
            elif body.startswith("$") or (len(out) >= 2 and out[-1] == "(" and out[-2] in _NAME_FUNCS):
                out.append(body)
            elif _ISO.match(body):
                out.append("<D>")
            elif _OID.match(body):
                out.append("<OID>")
            elif _UUID.match(body):
                out.append("<U>")
            else:
                out.append("<S>")
        else:
            out.append(v)
    return out


_QUOTED = re.compile(r"(?<![\w])(?:\"[^\"]*\"|'[^']*'|“[^”]*”|`[^`]*`)(?![\w])")
_IN_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:t\d{2}:\d{2}(?::\d{2})?z?)?\b")
_IN_OID = re.compile(r"\b[0-9a-f]{24}\b")
_IN_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")


def instr_tokens(text: str, literals: str = "placeholder") -> list[str]:
    """Instrucao em linguagem natural: sem acento, minuscula; valores citados/numeros/datas/ObjectIds -> placeholders."""
    t = bg.normalize_text(text or "")
    if literals == "placeholder":
        t = _QUOTED.sub(" <s> ", t)
        t = _IN_ISO.sub(" <d> ", t)
        t = _IN_OID.sub(" <oid> ", t)
        t = _IN_NUM.sub(" <n> ", t)
    return re.findall(r"<[a-z]+>|\w+", t)


# --------------------------------------------------------------------------- hashing / minhash
def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


_STRUCT = set("db getCollection getDatabase getConnection getSiblingDB ConnectionPool find findOne aggregate countDocuments distinct "
              "updateOne updateMany deleteOne deleteMany toArray sort limit skip project true false null new Date ISODate ObjectId "
              "UUID NumberLong NumberDecimal const let var _id createIndex dropIndex console log ENV get".split())
# Vocabulario da LINGUAGEM (argumentos de estagios do MQL e do Atlas Search): gramatica, nao conteudo de schema. Sem isso todo
# `$search` (`index/text/query/path`) teria >= 2 "tokens de conteudo" e a regra oficial removeria 100% de atlas_search.
_STRUCT |= set(("from localField foreignField as let pipeline path preserveNullAndEmptyArrays includeArrayIndex input cond then else "
                "branches case default groupBy boundaries output partitionBy sortBy window documents unit format timezone onNull "
                "index text query compound must mustNot should filter minimumShouldMatch fuzzy maxEdits prefixLength maxExpansions "
                "autocomplete tokenOrder sequential any phrase slop wildcard allowAnalyzedField near origin pivot equals "
                "range gt gte lt lte exists count type total highlight facet facets operator numBuckets lowerBound upperBound "
                "numCandidates limit queryVector exact searchScore vectorSearchScore geoWithin moreLikeThis queryString defaultPath "
                "synonyms scoreDetails returnStoredSource embedding").split())
_UPD = {"$set", "$inc", "$unset", "$currentDate", "$push", "$addToSet", "$pull", "$mul", "$min", "$max", "$rename", "$pop"}


def _known_ops() -> set[str]:
    try:
        import chat_validate as _cv
        return set(_cv.KNOWN_OPS) | set(_cv.ATLAS_OPS) | _UPD
    except Exception:  # pragma: no cover
        return set(_UPD) | {"$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$lookup", "$gt", "$gte", "$lt",
                            "$lte", "$in", "$nin", "$eq", "$ne", "$and", "$or", "$exists", "$regex", "$options", "$sum", "$avg"}


_OPS = _known_ops()
_PUNCT = re.compile(r"[^\w$]+")


def is_content(tok: str) -> bool:
    """True para token que carrega conteudo do schema (campo, colecao, constante nomeada); False para estrutura da linguagem
    (pontuacao, placeholders, operadores `$`, API do Console). `$campo` que nao e operador conta como conteudo."""
    if tok in _STRUCT or tok in _OPS or tok == "\x00":
        return False
    if tok.startswith("<") and tok.endswith(">"):
        return False
    return not _PUNCT.fullmatch(tok)


def ngram_map(tokens: list[str], n: int = NGRAM_N) -> dict[int, int]:
    """{hash do n-grama: numero de tokens de conteudo}."""
    if len(tokens) < n:
        return {}
    flags = [1 if is_content(t) else 0 for t in tokens]
    return {h64("".join(tokens[i:i + n])): sum(flags[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngrams(tokens: list[str], n: int = NGRAM_N) -> set[int]:
    return set(ngram_map(tokens, n))


class MinHasher:
    """MinHash proprio (sem libs): `num_perm` funcoes (a*h+b) mod 2^31-1 sobre o hash de 32 bits de cada shingle."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = 20261008):
        rng = random.Random(f"minhash:{seed}")
        self.ab = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num_perm)]
        self._np_ab = None

    def shingles(self, tokens: list[str], k: int = SHINGLE_K) -> set[int]:
        if not tokens:
            return set()
        if len(tokens) <= k:
            return {h64("\x1f".join(tokens)) & 0xFFFFFFFF}
        return {h64("\x1f".join(tokens[i:i + k])) & 0xFFFFFFFF for i in range(len(tokens) - k + 1)}

    def signature(self, tokens: list[str], k: int = SHINGLE_K) -> tuple[int, ...] | None:
        sh = self.shingles(tokens, k)
        if not sh:
            return None
        np = _numpy()
        if np is not None:
            # identico ao laco em Python: x < 2^32 e a < 2^31 => a*x + b < 2^63 (cabe em int64)
            if self._np_ab is None:
                self._np_ab = (np.array([a for a, _ in self.ab], dtype=np.int64), np.array([b for _, b in self.ab], dtype=np.int64))
            a, b = self._np_ab
            x = np.fromiter(sh, dtype=np.int64, count=len(sh))
            return tuple(((x[:, None] * a[None, :] + b[None, :]) % _P).min(axis=0).tolist())
        return tuple(min((a * x + b) % _P for x in sh) for a, b in self.ab)


def est_jaccard(a: tuple, b: tuple) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


class Lsh:
    """LSH por bandas sobre assinaturas MinHash; `query` devolve os ids com Jaccard estimado >= threshold."""

    def __init__(self, threshold: float, bands: int = BANDS, rows: int = ROWS):
        self.th, self.bands, self.rows = threshold, bands, rows
        self.tables: list[dict] = [dict() for _ in range(bands)]
        self.sigs: dict = {}

    def _keys(self, sig):
        r = self.rows
        return [sig[i * r:(i + 1) * r] for i in range(self.bands)]

    def add(self, key, sig):
        self.sigs[key] = sig
        for t, bk in zip(self.tables, self._keys(sig)):
            t.setdefault(bk, []).append(key)

    def query(self, sig, limit: int | None = None):
        cands = []
        seen = set()
        for t, bk in zip(self.tables, self._keys(sig)):
            for c in t.get(bk, ()):
                if c not in seen:
                    seen.add(c)
                    cands.append(c)
        hits = [(c, est_jaccard(sig, self.sigs[c])) for c in cands]
        hits = [(c, j) for c, j in hits if j >= self.th]
        hits.sort(key=lambda x: (-x[1], str(x[0])))
        return hits[:limit] if limit else hits


# --------------------------------------------------------------------------- features
_OP_RE = re.compile(r"\.(findOne|find|aggregate|countDocuments|distinct|updateOne|updateMany|deleteOne|deleteMany|insertOne|insertMany"
                    r"|replaceOne|createIndex|estimatedDocumentCount)\s*\(")
_COLL_RE = [re.compile(r"\bdb\.([A-Za-z_]\w*)\.(?:find|aggregate|count|distinct|update|delete|insert)"),
            re.compile(r"getCollection\(\s*['\"]([^'\"]+)['\"]\s*\)"),
            re.compile(r"ConnectionPool(?:\.\w+|\[['\"][^'\"]+['\"]\]){2}(?:\.|\[['\"])([A-Za-z_]\w*)")]


def record_kind(rec: dict) -> str:
    if "snippet_id" in rec and "instruction" in rec:
        return "chat"
    if "prefix" in rec and ("middle" in rec or "completion" in rec):
        return "fim"
    if "input" in rec and "output" in rec:
        return "legacy_chat"
    if "text" in rec:
        return "snippet"
    return "unknown"


def doc_text(rec: dict) -> tuple[str, str, str]:
    """(codigo, instrucao, editor) de um registro, qualquer formato conhecido (v1 incluido)."""
    k = record_kind(rec)
    if k == "fim":
        mid = rec["middle"] if "middle" in rec else rec.get("completion", "")
        return (rec.get("prefix") or "") + (mid or "") + (rec.get("suffix") or ""), "", ""
    if k == "chat":
        ctx = rec.get("context") or {}
        ed = ctx.get("editor") if isinstance(ctx, dict) else str(ctx or "")
        return rec.get("output") or "", rec.get("instruction") or "", ed or ""
    if k == "legacy_chat":
        cc = rec.get("chat_context") or {}
        return rec.get("output") or "", rec.get("input") or "", (cc.get("EditorContent") if isinstance(cc, dict) else "") or ""
    if k == "snippet":
        return rec.get("text") or "", "", ""
    return "", "", ""


def approx_key(code: str, collection: str | None = None) -> str | None:
    """Chave aproximada v1: operacao + colecao + conjunto de campos + conjunto de operadores (sem literais)."""
    m = _OP_RE.search(code)
    if not m:
        return None
    op = m.group(1)
    coll = collection
    if not coll:
        for rx in _COLL_RE:
            mm = rx.search(code)
            if mm:
                coll = mm.group(1)
                break
    # campos/operadores: tokens-chave (antes de ':') e referencias "$campo" -- reusa o tokenizador em modo keep
    fields: set[str] = set()
    ops: set[str] = set()
    pos, n = 0, len(code)
    while pos < n:
        mt = _CODE_RE.match(code, pos)
        pos = mt.end()
        k = mt.lastgroup
        v = mt.group()
        if k == "id":
            if code[pos:pos + 40].lstrip().startswith(":"):
                (ops if v.startswith("$") else fields).add(v)
        elif k == "str":
            body = _UNESC.sub(r"\1", v[1:-1])
            if code[pos:pos + 40].lstrip().startswith(":"):
                (ops if body.startswith("$") else fields).add(body)
            elif body.startswith("$") and not body.startswith("$$") and re.match(r"^\$[A-Za-z_][\w.]*$", body):
                fields.add(body[1:])
    return hashlib.sha1("|".join([op, (coll or "").lower(), ",".join(sorted(fields)), ",".join(sorted(ops))]).encode()).hexdigest()


def features(rec: dict, mh: MinHasher | None = None) -> dict:
    """Sinais de um registro para a decontaminacao (calculados uma vez por documento)."""
    code, instr, editor = doc_text(rec)
    ct = code_tokens(code)
    it = instr_tokens(instr)
    et = code_tokens(editor) if editor else []
    toks = ct + ["\x00"] + it + ["\x00"] + et if (it or et) else ct
    grams = ngram_map(ct)
    if it:
        grams.update(ngram_map(it))
    if et:
        grams.update(ngram_map(et))
    ak = approx_key(code, rec.get("collection") if isinstance(rec.get("collection"), str) else None)
    f = {
        "exact": hashlib.sha256(" ".join(toks).encode("utf-8")).hexdigest() if toks else None,
        "grams": grams,
        "approx": ak,
        "tokens": toks,
    }
    if mh is not None:
        f["sig"] = mh.signature(toks)
    return f


def doc_key(rec: dict) -> str:
    """Unidade de decisao: todos os cortes FIM do mesmo snippet compartilham o documento."""
    if record_kind(rec) == "fim" and rec.get("snippet_id"):
        return "doc:" + rec["snippet_id"]
    return "rec:" + str(rec.get("id"))


def dataset_hash(files: list[Path], extra: dict | None = None) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.name):
        fh = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        h.update(f"{p.name}:{fh.hexdigest()}\n".encode())
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- nucleo
def blind_status(blind=None) -> str:
    """`not_sealed` se nao ha manifesto selado (lido APENAS pelo blind_guard), senao `sealed`."""
    m = bg._manifest(blind)
    return "sealed" if (m.get("item_sha256") or m.get("prompt_sha256")) else "not_sealed"


def blind_hit(rec: dict, blind=None) -> bool:
    try:
        bg.check_record(rec, blind)
        code, instr, editor = doc_text(rec)
        if record_kind(rec) == "fim":
            bg.check_text((rec.get("prefix") or "") + "\x00" + (rec.get("suffix") or ""), blind)
        elif record_kind(rec) == "chat":
            bg.check_text(instr + "\x00" + editor, blind)
            bg.check_text(instr + "\x00", blind)
    except bg.BlindAccessError:
        return True
    return False


def build_index(eval_items: list[tuple[str, dict]], mh: MinHasher, threshold: float = DECONTAM_JACCARD) -> dict:
    """eval_items: [(origem, registro)]. Indexa por documento (FIM compartilha)."""
    idx = {"exact": {}, "grams": {}, "approx": {}, "lsh": Lsh(threshold), "n": 0, "by_source": {}}
    seen: set = set()
    for src, rec in eval_items:
        key = (src, doc_key(rec))
        if key in seen:
            continue
        seen.add(key)
        f = features(rec, mh)
        ident = f"{src}:{rec.get('id')}"
        idx["n"] += 1
        idx["by_source"][src] = idx["by_source"].get(src, 0) + 1
        if f["exact"]:
            idx["exact"].setdefault(f["exact"], ident)
        for g in f["grams"]:
            idx["grams"].setdefault(g, ident)
        if f["approx"]:
            idx["approx"].setdefault(f["approx"], ident)
        if f.get("sig"):
            idx["lsh"].add(ident, f["sig"])
    return idx


def match_item(f: dict, idx: dict, min_content: int = 0) -> dict[str, str]:
    """{regra: id_da_avaliacao} para um documento de treino. `_gram_best` = maior no. de tokens de conteudo entre os
    13-gramas comuns (sensibilidade); `ngram13` so conta se best >= min_content (0 = regra literal da spec)."""
    hit: dict[str, str] = {}
    if f["exact"] and f["exact"] in idx["exact"]:
        hit["exact"] = idx["exact"][f["exact"]]
    best, who = -1, None
    for g, c in f["grams"].items():
        if c > best:
            o = idx["grams"].get(g)
            if o:
                best, who = c, o
    if who is not None:
        hit["_gram_best"] = best
        if best >= min_content:
            hit["ngram13"] = who
    if f["approx"] and f["approx"] in idx["approx"]:
        hit["approx_key"] = idx["approx"][f["approx"]]
    if f.get("sig"):
        h = idx["lsh"].query(f["sig"], limit=1)
        if h:
            hit["minhash"] = h[0][0]
    return hit


def run(train: list[dict], evals: list[tuple[str, dict]], *, seed: int = 20261008, blind=None, threshold: float = DECONTAM_JACCARD,
        examples: int = 20, min_content: int | None = None, analysis_k: tuple = ()) -> dict:
    """Decontamina `train` contra `evals` [(origem, registro)]. Retorna dict com `removed` {id: [regras]} e contagens.

    `analysis_k` (ex.: (2,)): ANALISE paralela, nunca aplicada: quantos registros seriam removidos se o 13-grama exigisse
    >= k tokens de conteudo (as demais regras valem como estao); resultado em `analysis` {k: {removed, kept, by_cut_kind}}."""
    if min_content is None:
        min_content = DEFAULT_NGRAM_MIN_CONTENT
    mh = MinHasher(NUM_PERM, seed)
    idx = build_index(evals, mh, threshold)
    bstat = blind_status(blind)
    cache: dict[str, dict] = {}
    removed: dict[str, list[str]] = {}
    first: dict[str, str] = {}
    by_rule = {r: 0 for r in RULES}
    first_rule = {r: 0 for r in RULES}
    by_eval = {}
    ex: dict[str, list[str]] = {r: [] for r in RULES}
    by_kind: dict[str, int] = {}
    sens = {k: 0 for k in range(0, 5)}
    an = {int(k): {"removed": 0, "removed_by_cut_kind": {}, "in_by_cut_kind": {}} for k in analysis_k}
    for rec in train:
        dk = doc_key(rec)
        if dk not in cache:
            f = features(rec, mh)
            cache[dk] = match_item(f, idx, min_content)
        hit = dict(cache[dk])
        gb = hit.pop("_gram_best", None)
        if gb is not None:
            for k in range(0, 5):
                if gb >= k:
                    sens[k] += 1
        if bstat == "sealed" and blind_hit(rec, blind):
            hit["blind"] = "blind"
        if an:
            ck = rec.get("cut_kind") or rec.get("kind") or "?"
            other = any(r in hit for r in ("exact", "approx_key", "minhash", "blind"))
            for k, a in an.items():
                a["in_by_cut_kind"][ck] = a["in_by_cut_kind"].get(ck, 0) + 1
                if other or (gb is not None and gb >= k):
                    a["removed"] += 1
                    a["removed_by_cut_kind"][ck] = a["removed_by_cut_kind"].get(ck, 0) + 1
        if not hit:
            continue
        rules = [r for r in RULES if r in hit]
        removed[rec["id"]] = rules
        first[rec["id"]] = rules[0]
        first_rule[rules[0]] += 1
        for r in rules:
            by_rule[r] += 1
            if len(ex[r]) < examples:
                ex[r].append(rec["id"])
        for r, o in hit.items():
            src = o.split(":", 1)[0]
            by_eval[src] = by_eval.get(src, 0) + 1
        k = record_kind(rec)
        by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "blind": bstat,
        "train_in": len(train),
        "train_removed": len(removed),
        "train_kept": len(train) - len(removed),
        "removed": removed,
        "first_rule": first,
        "by_rule": by_rule,
        "by_rule_first_match": first_rule,
        "removed_by_kind": by_kind,
        "ngram13_sensitivity_min_content": {str(k): v for k, v in sens.items()},
        "analysis": {str(k): {**a, "kept": len(train) - a["removed"]} for k, a in an.items()},
        "matched_against": by_eval,
        "examples": ex,
        "eval_docs": idx["n"],
        "eval_docs_by_source": idx["by_source"],
        "params": {"ngram": NGRAM_N, "minhash_perm": NUM_PERM, "shingle_k": SHINGLE_K, "bands": BANDS, "rows": ROWS,
                   "jaccard": threshold, "seed": seed, "ngram_min_content": min_content},
    }


def public_report(res: dict, extra: dict) -> dict:
    """Relatorio sem o mapa completo de removidos (so contagens e <=20 ids de exemplo por regra)."""
    out = {k: v for k, v in res.items() if k not in ("removed", "first_rule")}
    out.update(extra)
    out["removed_ids_sha256"] = hashlib.sha256("\n".join(sorted(res["removed"])).encode()).hexdigest()
    return out


# --------------------------------------------------------------------------- CLI
HOMOLOG_DIR = ROOT / "data" / "homolog" / "v1"


def load_homolog(homolog_dir: Path | None = None) -> list[tuple[str, dict]]:
    """Registros do conjunto de homologacao (data/homolog/v1/*.jsonl, se existir) como referencia de avaliacao
    [("homolog", rec)]; entram na decontaminacao do treino com a MESMA regra oficial (13-grama, chave aproximada, MinHash)."""
    d = Path(homolog_dir) if homolog_dir else HOMOLOG_DIR
    out: list[tuple[str, dict]] = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if record_kind(r) != "unknown":
                        out.append(("homolog", r))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    bg.check_path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="jsonl com campo split (train/val/test/bench)")
    ap.add_argument("--legacy", nargs="*", default=[], help="jsonl legacy-v1 (somente avaliacao)")
    ap.add_argument("--eval-splits", nargs="+", default=list(EVAL_SPLITS))
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--ngram-min-content", type=int, default=DEFAULT_NGRAM_MIN_CONTENT,
                    help="N>0 (padrao 2) exige >= N tokens de conteudo (campos/colecoes) no 13-grama; 0 = regra literal da spec")
    ap.add_argument("--report-dir", default=str(ROOT / "reports" / "decontam"))
    ap.add_argument("--homolog", help="diretorio do conjunto de homologacao (padrao: data/homolog/v1 se existir)")
    ap.add_argument("--no-homolog", action="store_true", help="NAO decontaminar contra o conjunto de homologacao")
    ap.add_argument("--out-ids", help="grava os ids removidos (um por linha)")
    a = ap.parse_args(argv)
    files = [Path(p) for p in a.inputs]
    recs = [r for p in files for r in _read_jsonl(p)]
    train = [r for r in recs if r.get("split") == "train"]
    evals = [(r["split"], r) for r in recs if r.get("split") in set(a.eval_splits)]
    for p in a.legacy:
        evals += [("legacy-v1", r) for r in _read_jsonl(Path(p))]
    hom_guard.assert_no_hom(recs, "entradas do decontaminate (train/val/test/bench)")     # exit != 0 se houver schema hom no treino
    if not a.no_homolog:
        evals += load_homolog(a.homolog)
    res = run(train, evals, seed=a.seed, min_content=a.ngram_min_content)
    dh = dataset_hash(files + [Path(p) for p in a.legacy], {"seed": a.seed, "params": res["params"]})
    rep = public_report(res, {"dataset_hash": dh, "inputs": [str(p) for p in files], "legacy": a.legacy})
    out = Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dh[:16]}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.out_ids:
        Path(a.out_ids).write_text("\n".join(sorted(res["removed"])) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("blind", "train_in", "train_removed", "by_rule", "dataset_hash")}))


if __name__ == "__main__":
    main()
