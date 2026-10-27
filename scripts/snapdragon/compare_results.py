"""Compare complete inference reports on the same frozen sample (no code execution)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from collections import Counter


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (rank - low)


def summarize(report, corpus, tokenizer):
    expected = {row["id"]: row for row in corpus}
    actual = {row["id"]: row for row in report.get("items", [])}
    if len(actual) != len(report.get("items", [])) or set(actual) != set(expected):
        raise ValueError("Report IDs must match the complete corpus exactly, without duplicates")
    if report.get("status") != "completed" or report.get("error") or any(row.get("error") for row in actual.values()):
        raise ValueError("Incomplete or failed runs cannot produce quality acceptance")
    prefix = []
    exact = []
    ttft = []
    speed = []
    stops = ("<|endoftext|>", "<|im_end|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>")
    for key, row in actual.items():
        sample = expected[key]
        reference = next((sample[k] for k in ("reference", "completion", "target", "middle") if isinstance(sample.get(k), str)), None)
        if reference is None:
            raise ValueError(f"Missing reference: {key}")
        text = row["generated_text"]
        for stop in stops:
            text = text.split(stop)[0]
        target_ids = tokenizer.encode(reference, add_special_tokens=False).ids
        actual_ids = tokenizer.encode(text, add_special_tokens=False).ids
        shared = 0
        for a, b in zip(target_ids, actual_ids):
            if a != b:
                break
            shared += 1
        prefix.append(shared)
        exact.append(text.strip() == reference.strip())
        if row.get("ttft_ms") is not None:
            ttft.append(row["ttft_ms"])
        if row.get("decode_tokens_per_second") is not None:
            speed.append(row["decode_tokens_per_second"])
    return {"n": len(actual), "accepted_prefix_tokens": statistics.mean(prefix),
            "trimmed_exact_match": statistics.mean(exact), "ttft_p50_ms": percentile(ttft, .5),
            "ttft_p95_ms": percentile(ttft, .95), "decode_tokens_per_second_median": statistics.median(speed) if speed else None,
            "ttft_observations": len(ttft), "decode_rate_observations": len(speed),
            "stop_reasons": dict(Counter(row.get("stop_reason", "not_recorded") for row in actual.values())),
            "observed_eos_count": sum(bool(row.get("eos")) for row in actual.values()),
            "generated_tokens_total": sum(len(row.get("generated_ids", [])) for row in actual.values()),
            "provider_evidence": report.get("provider_evidence"), "memory": report.get("memory_after")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("corpus", "baseline", "candidate", "tokenizer", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    corpus = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["id"] for row in corpus}) != len(corpus):
        raise ValueError("Duplicate corpus IDs")
    corpus_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.baseline, args.candidate)]
    if any(report.get("corpus_sha256") != corpus_hash for report in reports):
        raise ValueError("Both reports must record the SHA-256 of this exact corpus")
    baseline, candidate = [summarize(report, corpus, tok) for report in reports]
    result = {"schema": "kapi-snapdragon-comparison-v1", "qualified": False, "proxy": True,
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              "baseline": baseline, "candidate": candidate,
              "delta_accepted_prefix_tokens": candidate["accepted_prefix_tokens"] - baseline["accepted_prefix_tokens"],
              "limitations": ["Small synthetic sample; no IDE qualification", "Generated JavaScript was not executed", "Decode rate uses only recorded non-null observations; EOS may be omitted by GenAI"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
