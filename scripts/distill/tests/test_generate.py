"""Determinismo / retomada / seeds do motor de geracao (backend falso: sem GPU nem torch)."""
import json
from pathlib import Path

import pytest

import generate_teacher as gt
from _distill_common import item_seed

STOP = 0


class FakeBackend:
    """Deterministico em (id, k): tokens derivados do hash; EOS depois de n tokens."""
    stops = frozenset({STOP})

    def __init__(self):
        self.calls = 0

    def encode(self, rec):
        return list(range(1, 1 + len(rec["prefix"])))

    def decode(self, tokens):
        return "".join(chr(97 + (t % 26)) for t in tokens)

    def postprocess(self, text, rec):
        return text

    def generate(self, batch, *, seeds, max_new, temperature, top_p, topk_k):
        self.calls += 1
        cands = []
        for it, sd in zip(batch, seeds):
            row = []
            for k in range(1 + len(sd)):
                base = item_seed(7, it["id"], k)
                n = 2 + base % 5
                toks = [1 + (base >> (3 * i)) % 25 for i in range(n)] + [STOP]
                c = {"k": k, "mode": "greedy" if k == 0 else "sample", "seed": None if k == 0 else sd[k - 1],
                     "tokens": toks, "logprobs": [-0.1 * (i + 1) for i in range(len(toks))]}
                if topk_k:
                    import numpy as np
                    c["topk"] = {"ids": np.arange(len(toks) * topk_k, dtype=np.uint32).reshape(len(toks), topk_k),
                                 "lps": np.zeros((len(toks), topk_k), dtype=np.float16), "tail": np.zeros(len(toks), dtype=np.float16)}
                row.append(c)
            cands.append(row)
        return {"cands": cands, "stats": {}}


def make_pool(tmp: Path, n=9) -> Path:
    pool = tmp / "pool"
    pool.mkdir()
    rows = [{"id": f"r{i}", "prefix": "p" * (1 + (i * 7) % 11), "suffix": "s", "gold": "SEGREDO", "origin": "base"} for i in range(n)]
    for s in range(0, n, 5):
        with open(pool / f"pool-{s // 5:05d}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows[s:s + 5]:
                f.write(json.dumps(r) + "\n")
    return pool


def engine(tmp, out, backend, **kw):
    args = dict(block_size=2, shard_size=4, max_new=8, samples=1, temperature=0.2, top_p=1.0, topk_k=4, save_topk=False, base_seed=7,
                batch_tokens=40, max_rows=3, log=lambda *_: None)
    args.update(kw)
    return gt.Engine(backend, make_pool(tmp) if not (tmp / "pool").exists() else tmp / "pool", out, **args)


def read_all(out: Path):
    rows = []
    for p in sorted(out.glob("gen-*.jsonl")):
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return rows


def test_seed_depends_on_id_not_position():
    assert item_seed(1, "abc", 1) == item_seed(1, "abc", 1)
    assert item_seed(1, "abc", 1) != item_seed(1, "abc", 2)
    assert item_seed(1, "abc", 1) != item_seed(2, "abc", 1)


def test_full_run_is_deterministic_and_has_greedy_plus_sample(tmp_path):
    a = engine(tmp_path, tmp_path / "a", FakeBackend())
    a.run()
    ra = read_all(tmp_path / "a")
    assert len(ra) == 9 * 2 and {r["k"] for r in ra} == {0, 1}
    assert {r["mode"] for r in ra} == {"greedy", "sample"}
    assert all(r["seed"] == item_seed(7, r["id"], 1) for r in ra if r["k"] == 1)
    assert all("SEGREDO" not in json.dumps(r) for r in ra)            # o gold nunca chega ao professor/saida de geracao
    b = engine(tmp_path, tmp_path / "b", FakeBackend())
    b.run()
    assert (tmp_path / "a" / "gen-00000.jsonl").read_bytes() == (tmp_path / "b" / "gen-00000.jsonl").read_bytes()


def test_resume_after_crash_gives_identical_output(tmp_path):
    full = engine(tmp_path, tmp_path / "full", FakeBackend())
    full.run()

    class Boom(Exception):
        pass

    def hook(blk, n):
        if blk == 2:
            raise Boom

    crashed = engine(tmp_path, tmp_path / "res", FakeBackend())
    with pytest.raises(Boom):
        crashed.run(hook=hook)
    done_lines = (tmp_path / "res" / "done.idx").read_text().splitlines()
    assert [int(l.split()[0]) for l in done_lines] == [0, 1]
    # simula morte no meio da escrita do bloco 2: bytes extras no shard + linha rasgada no done.idx
    shard = tmp_path / "res" / "gen-00000.jsonl"
    with open(shard, "ab") as f:
        f.write(b'{"idx": 4, "id": "r4", "k": 0, "raw": "lixo')
    with open(tmp_path / "res" / "done.idx", "a") as f:
        f.write("2 4 5 4 gen-00001.jsonl")
    resumed = engine(tmp_path, tmp_path / "res", FakeBackend())
    resumed.run(resume=True)
    for name in ("gen-00000.jsonl", "gen-00001.jsonl", "gen-00002.jsonl"):
        assert (tmp_path / "res" / name).read_bytes() == (tmp_path / "full" / name).read_bytes()
    rows = read_all(tmp_path / "res")
    assert len({(r["idx"], r["k"]) for r in rows}) == len(rows)       # sem duplicatas


def test_resume_skips_committed_blocks(tmp_path):
    e1 = engine(tmp_path, tmp_path / "o", FakeBackend())
    e1.run()
    fb = FakeBackend()
    e2 = engine(tmp_path, tmp_path / "o", fb)
    e2.run(resume=True)
    assert fb.calls == 0


def test_batches_depend_only_on_block_content(tmp_path):
    e = engine(tmp_path, tmp_path / "o", FakeBackend(), block_size=5, shard_size=5)
    items = [{"idx": i, "id": f"r{i}", "ids": [0] * n} for i, n in enumerate([5, 3, 9, 3, 7])]
    b1 = [[it["idx"] for it in b] for b in e.make_batches(items)]
    b2 = [[it["idx"] for it in b] for b in e.make_batches(list(reversed(items)))]
    assert b1 == b2
    assert all(len(b) <= 3 for b in b1)


def test_topk_shard_and_cost(tmp_path):
    import numpy as np
    e = engine(tmp_path, tmp_path / "o", FakeBackend(), save_topk=True, topk_k=4)
    s = e.run()
    parts = sorted((tmp_path / "o" / "topk").glob("t-*.npz"))
    assert parts
    z = np.load(parts[0])
    assert z["topk_ids"].dtype == np.uint32 and z["topk_logprobs"].dtype == np.float16 and z["tail_mass"].dtype == np.float16
    assert z["topk_ids"].shape[1] == 4 and int(z["n_pos"].sum()) == z["topk_ids"].shape[0]
    assert s["topk_bytes_per_position_theoretical"] == 4 * 4 + 4 * 2 + 2
    assert gt.TOPK_BYTES_PER_POS(32) == 194
