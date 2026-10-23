#!/usr/bin/env python3
"""Proxy TCP SOMENTE-LEITURA para o protocolo de rede do MongoDB (so biblioteca padrao; roda no WSL junto do mongosh).

Fica entre o mongosh e o mongod e e a PROVA de leitura: cada mensagem OP_MSG do cliente e inspecionada ANTES de chegar ao servidor.
  * allowlist de comandos (find, aggregate, count, distinct, getMore, killCursors, explain de leitura, listCollections, listIndexes,
    hello/isMaster/ping/buildInfo/endSessions/whatsmyuri/...); qualquer outro (insert, update, delete, createIndexes, drop, ...) e
    RECUSADO com {ok:0, code:13} e contado em `blocked`;
  * aggregate com $out/$merge (e $function/$accumulator/$where) e recusado; comandos de dados so em bancos permitidos (kapi_fx_*);
  * injeta `maxTimeMS` (padrao 3000) em find/aggregate/count/distinct que nao o tragam; aplica teto se trouxerem;
  * contadores por comando (permitidos/recusados), bytes e erros do servidor; registra os comandos de leitura (BSON base64) por consulta
    (marcador: ping com `comment: "KAPI:S:<id>"`) e a contagem de documentos devolvidos (firstBatch/nextBatch/n) para o validador.
Sem compressao (use compressors=disabled na URI). OP_QUERY legado so para o hello inicial.
"""
from __future__ import annotations

import base64
import json
import socket
import struct
import threading

OP_MSG, OP_QUERY, OP_REPLY, OP_COMPRESSED = 2013, 2004, 1, 2012
DATA_CMDS = {"find", "aggregate", "count", "distinct"}
CURSOR_CMDS = {"getmore", "killcursors"}
META_READ = {"listcollections", "listindexes", "dbstats", "collstats"}
HANDSHAKE = {"hello", "ismaster", "ping", "buildinfo", "endsessions", "whatsmyuri", "atlasversion", "connectionstatus", "getparameter",
             "getcmdlineopts", "hostinfo", "listdatabases", "serverstatus", "features", "replsetgetstatus", "saslstart", "saslcontinue"}
ALLOWED = DATA_CMDS | CURSOR_CMDS | META_READ | HANDSHAKE | {"explain"}
BAD_AGG = (b"$out\x00", b"$merge\x00", b"$function\x00", b"$accumulator\x00", b"$where\x00", b"$currentOp\x00", b"$listSessions\x00")


# ---------------------------------------------------------------------------------------------------------------- BSON minimo
def _cstr(b: bytes, i: int):
    j = b.index(b"\x00", i)
    return b[i:j].decode("utf-8", "replace"), j + 1


def _skip(t: int, b: bytes, i: int) -> int:
    if t == 0x01 or t in (0x09, 0x11, 0x12):
        return i + 8
    if t in (0x02, 0x0D, 0x0E):
        return i + 4 + struct.unpack_from("<i", b, i)[0]
    if t in (0x03, 0x04):
        return i + struct.unpack_from("<i", b, i)[0]
    if t == 0x05:
        return i + 5 + struct.unpack_from("<i", b, i)[0]
    if t in (0x06, 0x0A, 0xFF, 0x7F):
        return i
    if t == 0x07:
        return i + 12
    if t == 0x08:
        return i + 1
    if t == 0x0B:
        _, i = _cstr(b, i)
        _, i = _cstr(b, i)
        return i
    if t == 0x0C:
        return i + 4 + struct.unpack_from("<i", b, i)[0] + 12
    if t == 0x0F:
        return i + struct.unpack_from("<i", b, i)[0]
    if t == 0x10:
        return i + 4
    if t == 0x13:
        return i + 16
    raise ValueError(f"tipo BSON {t:#x}")


def elements(doc: bytes):
    """(tipo, nome, ini_valor, fim_valor, ini_elemento) dos elementos do nivel superior de um documento BSON."""
    n = struct.unpack_from("<i", doc, 0)[0]
    i = 4
    while i < n - 1:
        t = doc[i]
        name, j = _cstr(doc, i + 1)
        k = _skip(t, doc, j)
        yield t, name, j, k, i
        i = k


def first_key(doc: bytes) -> str | None:
    for t, name, *_ in elements(doc):
        return name
    return None


def get(doc: bytes, key: str):
    for t, name, a, b, s in elements(doc):
        if name == key:
            return t, doc[a:b]
    return None


def get_str(doc: bytes, key: str) -> str | None:
    r = get(doc, key)
    if r and r[0] == 0x02:
        n = struct.unpack_from("<i", r[1], 0)[0]
        return r[1][4:4 + n - 1].decode("utf-8", "replace")
    return None


def get_num(doc: bytes, key: str):
    r = get(doc, key)
    if not r:
        return None
    t, v = r
    if t == 0x10:
        return struct.unpack("<i", v)[0]
    if t == 0x12:
        return struct.unpack("<q", v)[0]
    if t == 0x01:
        return struct.unpack("<d", v)[0]
    return None


def add_max_time(doc: bytes, ms: int) -> bytes:
    """Acrescenta maxTimeMS (int64) ao fim do documento (antes do terminador); ajusta o comprimento."""
    body = doc[4:-1] + b"\x12maxTimeMS\x00" + struct.pack("<q", ms)
    return struct.pack("<i", len(body) + 5) + body + b"\x00"


def bson_doc(pairs) -> bytes:
    out = b""
    for k, v in pairs:
        if isinstance(v, bool):
            out += b"\x08" + k.encode() + b"\x00" + (b"\x01" if v else b"\x00")
        elif isinstance(v, int):
            out += b"\x10" + k.encode() + b"\x00" + struct.pack("<i", v)
        elif isinstance(v, float):
            out += b"\x01" + k.encode() + b"\x00" + struct.pack("<d", v)
        else:
            s = str(v).encode()
            out += b"\x02" + k.encode() + b"\x00" + struct.pack("<i", len(s) + 1) + s + b"\x00"
    return struct.pack("<i", len(out) + 5) + out + b"\x00"


def op_msg_reply(request_id: int, resp_to: int, doc: bytes) -> bytes:
    body = struct.pack("<I", 0) + b"\x00" + doc
    return struct.pack("<iiii", 16 + len(body), request_id, resp_to, OP_MSG) + body


def parse_op_msg(payload: bytes):
    """-> (flags, body_doc, tem_sequencias). payload comeca nos flagBits."""
    flags = struct.unpack_from("<I", payload, 0)[0]
    end = len(payload) - (4 if flags & 1 else 0)
    i, body, seqs = 4, None, False
    while i < end:
        kind = payload[i]
        i += 1
        if kind == 0:
            n = struct.unpack_from("<i", payload, i)[0]
            body = payload[i:i + n]
            i += n
        elif kind == 1:
            n = struct.unpack_from("<i", payload, i)[0]
            seqs = True
            i += n
        else:
            raise ValueError("secao desconhecida")
    return flags, body, seqs


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.allowed: dict = {}
        self.blocked: dict = {}
        self.server_errors = 0
        self.docs_by_cmd: dict = {}
        self.bytes_c2s = 0
        self.bytes_s2c = 0

    def add(self, table, name):
        with self.lock:
            table[name] = table.get(name, 0) + 1

    def to_dict(self):
        with self.lock:
            return {"allowed": dict(self.allowed), "blocked": dict(self.blocked), "server_errors": self.server_errors,
                    "bytes_c2s": self.bytes_c2s, "bytes_s2c": self.bytes_s2c,
                    "writes_seen_by_server": {k: v for k, v in self.allowed.items() if k in ("insert", "update", "delete", "findandmodify",
                                                                                           "createindexes", "drop", "dropdatabase", "create")}}


class RoProxy:
    def __init__(self, upstream=("127.0.0.1", 27017), allowed_dbs=None, max_time_ms=3000, event_cb=None, listen_host="127.0.0.1"):
        self.upstream = upstream
        self.allowed_dbs = set(allowed_dbs or [])
        self.max_time_ms = max_time_ms
        self.stats = Stats()
        self.event_cb = event_cb or (lambda ev: None)
        self.cur_id = threading.local()
        self.global_cur = None            # id da consulta em curso (uma conexao de script por vez)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_host, 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.stop = False
        self.threads = []

    def serve_forever(self):
        t = threading.Thread(target=self._accept, daemon=True)
        t.start()
        return t

    def _accept(self):
        while not self.stop:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(c,), daemon=True).start()

    def close(self):
        self.stop = True
        try:
            self.sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------------------------
    def _recv_msg(self, s):
        hdr = b""
        while len(hdr) < 16:
            ch = s.recv(16 - len(hdr))
            if not ch:
                return None
            hdr += ch
        n = struct.unpack_from("<i", hdr, 0)[0]
        buf = bytearray(hdr)
        while len(buf) < n:
            ch = s.recv(min(65536, n - len(buf)))
            if not ch:
                return None
            buf += ch
        return bytes(buf)

    def _conn(self, c):
        try:
            u = socket.create_connection(self.upstream, timeout=10)
        except OSError:
            c.close()
            return
        u.settimeout(None)
        pending: list = []          # (comando, id da consulta) por requisicao, na ordem (resposta ocorre na mesma ordem por conexao)
        try:
            while True:
                msg = self._recv_msg(c)
                if msg is None:
                    break
                self.stats.bytes_c2s += len(msg)
                verdict = self.inspect(msg)
                if verdict[0] == "block":
                    req_id = struct.unpack_from("<i", msg, 4)[0]
                    reply = op_msg_reply(0, req_id, bson_doc([("ok", 0.0), ("errmsg", "kapi read-only proxy: " + verdict[1]), ("code", 13),
                                                              ("codeName", "Unauthorized")]))
                    c.sendall(reply)
                    continue
                out_msg, cmd, qid = verdict[1], verdict[2], verdict[3]
                u.sendall(out_msg)
                flags = struct.unpack_from("<I", out_msg, 16)[0] if struct.unpack_from("<i", out_msg, 12)[0] == OP_MSG else 0
                if struct.unpack_from("<i", out_msg, 12)[0] == OP_MSG and (flags & 2):       # moreToCome: sem resposta
                    continue
                resp = self._recv_msg(u)
                if resp is None:
                    break
                self.stats.bytes_s2c += len(resp)
                self.on_reply(resp, cmd, qid)
                c.sendall(resp)
                # respostas "exhaust" (moreToCome do servidor): repassa ate acabar
                while struct.unpack_from("<i", resp, 12)[0] == OP_MSG and (struct.unpack_from("<I", resp, 16)[0] & 2):
                    resp = self._recv_msg(u)
                    if resp is None:
                        break
                    c.sendall(resp)
        except (OSError, ValueError, struct.error):
            pass
        finally:
            for s in (c, u):
                try:
                    s.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------------------------------------------------
    def inspect(self, msg: bytes):
        op = struct.unpack_from("<i", msg, 12)[0]
        if op == OP_QUERY:
            # flags(4) cstring ns, skip(4) return(4), doc
            i = 16 + 4
            ns, i = _cstr(msg, i)
            i += 8
            doc = msg[i:i + struct.unpack_from("<i", msg, i)[0]]
            name = (first_key(doc) or "").lower()
            if name in ("ismaster", "hello") and ns.endswith(".$cmd"):
                self.stats.add(self.stats.allowed, name)
                return ("pass", msg, name, None)
            self.stats.add(self.stats.blocked, "op_query:" + name)
            return ("block", "OP_QUERY recusado")
        if op != OP_MSG:
            self.stats.add(self.stats.blocked, f"opcode:{op}")
            return ("block", f"opcode {op} recusado")
        payload = msg[16:]
        flags, body, seqs = parse_op_msg(payload)
        if body is None:
            return ("block", "sem corpo")
        raw_name = first_key(body) or ""
        name = raw_name.lower()
        db = get_str(body, "$db")
        if name not in ALLOWED:
            self.stats.add(self.stats.blocked, name)
            return ("block", f"comando {raw_name} nao permitido")
        if seqs:
            self.stats.add(self.stats.blocked, name + ":sequencia")
            return ("block", "sequencia de documentos nao permitida")
        if name in DATA_CMDS | CURSOR_CMDS | META_READ | {"explain"}:
            if self.allowed_dbs and db not in self.allowed_dbs:
                self.stats.add(self.stats.blocked, f"{name}:db={db}")
                return ("block", f"banco {db} fora da lista permitida")
        if name == "explain":
            inner = get(body, "explain")
            if not inner or inner[0] != 0x03 or (first_key(inner[1]) or "").lower() not in DATA_CMDS:
                self.stats.add(self.stats.blocked, "explain:nao_leitura")
                return ("block", "explain so de leitura")
            if any(b in inner[1] for b in BAD_AGG):
                self.stats.add(self.stats.blocked, "explain:agg_proibido")
                return ("block", "operador proibido")
        if name == "aggregate":
            if any(b in body for b in BAD_AGG):
                self.stats.add(self.stats.blocked, "aggregate:$out/$merge/$function")
                return ("block", "aggregate com $out/$merge/$function recusado")
        qid = self.global_cur
        if name == "ping":
            cm = get_str(body, "comment")
            if cm and cm.startswith("KAPI:S:"):
                self.global_cur = cm[7:]
                self.event_cb({"ev": "start", "id": self.global_cur})
            elif cm and cm.startswith("KAPI:E:"):
                self.event_cb({"ev": "end", "id": cm[7:]})
                self.global_cur = None
        out = msg
        if name in DATA_CMDS:
            mt = get_num(body, "maxTimeMS")
            if mt is not None and (mt > self.max_time_ms or mt <= 0):
                self.stats.add(self.stats.blocked, name + ":maxTimeMS_alto")
                return ("block", "maxTimeMS acima do teto")
            if mt is None:
                new_payload = payload[:4] + b"\x00" + add_max_time(body, self.max_time_ms) + payload[5 + len(body):]
                out = struct.pack("<iiii", 16 + len(new_payload), *struct.unpack_from("<iii", msg, 4)) + new_payload
            self.event_cb({"ev": "cmd", "id": qid, "name": raw_name, "db": db, "bson": base64.b64encode(body).decode()})
        self.stats.add(self.stats.allowed, name)
        return ("pass", out, name, qid)

    def on_reply(self, resp: bytes, cmd: str, qid):
        try:
            if struct.unpack_from("<i", resp, 12)[0] != OP_MSG:
                return
            _, body, _ = parse_op_msg(resp[16:])
            if body is None:
                return
            ok = get_num(body, "ok")
            if ok is not None and ok == 0:
                with self.stats.lock:
                    self.stats.server_errors += 1
                self.event_cb({"ev": "err", "id": qid, "cmd": cmd, "code": get_num(body, "code"), "errmsg": (get_str(body, "errmsg") or "")[:300]})
                return
            if cmd in DATA_CMDS | {"getmore"}:
                n = None
                cur = get(body, "cursor")
                if cur and cur[0] == 0x03:
                    for key in ("firstBatch", "nextBatch"):
                        arr = get(cur[1], key)
                        if arr and arr[0] == 0x04:
                            n = sum(1 for _ in elements(arr[1]))
                elif cmd == "count":
                    n = get_num(body, "n")
                elif cmd == "distinct":
                    arr = get(body, "values")
                    n = sum(1 for _ in elements(arr[1])) if arr and arr[0] == 0x04 else None
                self.event_cb({"ev": "docs", "id": qid, "cmd": cmd, "n": n})
        except (ValueError, struct.error, IndexError):
            pass


if __name__ == "__main__":
    import sys
    px = RoProxy(allowed_dbs=set(sys.argv[1:]) or None, event_cb=lambda e: print(json.dumps(e)[:200], flush=True))
    print("porta", px.port, flush=True)
    px.serve_forever().join()
