"""Testes do guard e da ferramenta do conjunto cego.

  env\\.venv\\Scripts\\python.exe -m pytest scripts/data/test_blind_guard.py
  env\\.venv\\Scripts\\python.exe -m unittest scripts.data.test_blind_guard   (sem pytest)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard as G  # noqa: E402
import blind_tool as T  # noqa: E402


def item(n=1, **kw):
    d = {"id": f"blind-autocomplete-{n:03d}", "kind": "autocomplete", "language": "en", "domain": "legal",
         "schema_id": "juris", "collection": "Cases", "prefix": f"db.Cases.find({{ Status: {n}", "suffix": " })",
         "references": ['"open"'], "author": "human", "reviewed_by": "alice", "status": "reviewed"}
    d.update(kw)
    return d


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.blind = self.root / "data" / "eval" / "blind"
        (self.blind / "items").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_blocks_inside(self):
        for p in (self.blind, self.blind / "items" / "items.jsonl", self.blind / "nao" / "existe.txt",
                  self.blind / "items" / ".." / "items" / "x"):
            with self.assertRaises(G.BlindAccessError):
                G.check_path(p, self.blind)

    def test_allows_outside_and_lookalike(self):
        for p in (self.root / "data" / "eval", self.root / "data" / "eval" / "blind_other" / "x",
                  self.root / "data" / "generated" / "a.jsonl", self.blind.parent / "legacy-v1" / "x"):
            self.assertEqual(G.check_path(p, self.blind), p)

    def test_blocks_dotdot_escape_into_blind(self):
        p = self.root / "data" / "generated" / ".." / "eval" / "blind" / "items"
        with self.assertRaises(G.BlindAccessError):
            G.check_path(p, self.blind)

    def test_env_override(self):
        os.environ["KAPI_BLIND_DIR"] = str(self.blind)
        try:
            with self.assertRaises(G.BlindAccessError):
                G.check_path(self.blind / "a")
            G.check_path(self.root / "a")
        finally:
            del os.environ["KAPI_BLIND_DIR"]

    def test_default_location_is_repo_data_eval_blind(self):
        with self.assertRaises(G.BlindAccessError):
            G.check_path(G.ROOT / "data" / "eval" / "blind" / "items" / "items.jsonl")

    def test_hash_guards(self):
        it = item()
        T.main(["--blind-dir", str(self.blind), "init"])
        f = self.root / "in.jsonl"
        f.write_text(json.dumps(it) + "\n", encoding="utf-8")
        T.main(["--blind-dir", str(self.blind), "add", "--file", str(f)])
        T.main(["--blind-dir", str(self.blind), "seal"])
        with self.assertRaises(G.BlindAccessError):
            G.check_record(it, self.blind)
        with self.assertRaises(G.BlindAccessError):
            G.check_text(G.prompt_text(it).upper().replace("\x00", "\x00 "), self.blind)
        G.check_text("outra coisa", self.blind)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.blind = self.root / "blind"
        self.base = ["--blind-dir", str(self.blind), "--who", "tester"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = T.main(self.base + list(args)) or 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def jsonl(self, name, items):
        p = self.root / name
        p.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
        return str(p)

    def test_full_flow(self):
        self.assertEqual(self.run_cli("init")[0], 0)
        self.assertTrue((self.blind / "items").is_dir() and (self.blind / "sealed").is_dir())
        self.assertTrue((self.blind / "access.log.jsonl").exists())
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("a.jsonl", [item(1), item(2, language="pt-BR")]))[0], 0)
        self.assertEqual(self.run_cli("verify")[0], 2)  # nao selado
        self.assertEqual(self.run_cli("seal")[0], 0)
        self.assertEqual(self.run_cli("verify")[0], 0)
        # add recusado apos selar
        c, _, e = self.run_cli("add", "--file", self.jsonl("b.jsonl", [item(3)]))
        self.assertEqual(c, 3, e)
        # adulteracao detectada
        f = self.blind / "items" / "items.jsonl"
        txt = f.read_text(encoding="utf-8")
        f.write_text(txt.replace('open', 'shut', 1), encoding="utf-8")
        self.assertEqual(self.run_cli("verify")[0], 1)
        f.write_text(txt, encoding="utf-8")
        self.assertEqual(self.run_cli("verify")[0], 0)
        # nova rodada
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("c.jsonl", [item(3)]), "--new-round")[0], 2)  # sem reason
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("c.jsonl", [item(3)]), "--new-round", "--reason", "rodada 2")[0], 0)
        self.assertEqual(self.run_cli("verify")[0], 2)  # manifesto arquivado; precisa selar de novo
        self.assertEqual(self.run_cli("seal")[0], 0)
        m = json.loads((self.blind / "sealed" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((m["round"], m["count"], m["version"]), (2, 3, "blind-v2"))

    def test_add_validation(self):
        self.run_cli("init")
        bad = item(1)
        del bad["references"]
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("x.jsonl", [item(2), bad]))[0], 1)
        self.assertEqual(self.run_cli("stats", "--json")[0], 0)
        self.assertEqual(json.loads(self.run_cli("stats", "--json")[1])["count"], 0)  # atomico: nada gravado
        for mutate in ({"kind": "other"}, {"language": "fr"}, {"schema_id": "juris", "collection": "Nope"},
                       {"domain": "finance"}, {"status": "draft", "reviewed_by": "bob"},
                       {"status": "reviewed", "reviewed_by": None}, {"extra": 1}, {"prefix": "  "}):
            code = self.run_cli("add", "--file", self.jsonl("y.jsonl", [item(5, **mutate)]))[0]
            self.assertEqual(code, 1, mutate)
        # chat exige instruction; correction exige context
        chat = item(7, kind="chat", id="blind-chat-007")
        chat.pop("prefix"), chat.pop("suffix")
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("z.jsonl", [chat]))[0], 1)
        chat["instruction"] = "liste os casos abertos"
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("z.jsonl", [chat]))[0], 0)
        corr = dict(chat, id="blind-correction-001", kind="correction", instruction="corrija")
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("w.jsonl", [corr]))[0], 1)
        # id duplicado
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("d.jsonl", [chat]))[0], 1)

    def test_seal_refuses_drafts(self):
        self.run_cli("init")
        d = item(1, author="draft-agent", status="draft", reviewed_by=None)
        self.assertEqual(self.run_cli("add", "--file", self.jsonl("d.jsonl", [d]))[0], 0)
        self.assertEqual(self.run_cli("seal")[0], 1)
        self.assertFalse((self.blind / "sealed" / "manifest.json").exists())

    def test_read_requires_reason_and_logs(self):
        self.run_cli("init")
        self.run_cli("add", "--file", self.jsonl("a.jsonl", [item(1)]))
        self.assertEqual(self.run_cli("read")[0], 2)
        c, out, _ = self.run_cli("read", "--reason", "gate final T1", "--model", "T1-run3")
        self.assertEqual(c, 0)
        self.assertEqual(json.loads(out.splitlines()[0])["id"], "blind-autocomplete-001")
        recs = [json.loads(l) for l in (self.blind / "access.log.jsonl").read_text(encoding="utf-8").splitlines()]
        r = [x for x in recs if x["action"] == "read"][-1]
        self.assertEqual((r["who"], r["reason"], r["model"], r["n_items"]), ("tester", "gate final T1", "T1-run3", 1))
        self.assertIn("ts", r)

    def test_stats_checks(self):
        self.run_cli("init")
        its = [item(1, domain="education", schema_id="campus", collection="students"),
               item(2, language="pt-BR"), item(3, language="pt-BR", domain="finance", schema_id="ledgerly",
                                                collection="accounts"),
               item(4)]
        self.run_cli("add", "--file", self.jsonl("a.jsonl", its))
        s = json.loads(self.run_cli("stats", "--json")[1])
        checks = {c["check"]: c["ok"] for c in s["checks"]}
        self.assertTrue(checks["holdout>=40%"])  # 2/4 holdout
        self.assertTrue(checks["pt-BR x en balanceado (45-55%)"])
        self.assertFalse(checks["tamanho 400"])
        self.assertEqual(self.run_cli("stats", "--strict")[0], 1)


if __name__ == "__main__":
    unittest.main()
