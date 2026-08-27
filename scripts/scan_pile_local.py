#!/usr/bin/env python3
import os
import re
import json
import argparse
from itertools import islice

# IMPORTANT: set caches BEFORE importing datasets.
# Anchor on this file's location (<root>/scripts/), not the CWD, so the cache
# resolves the same whether run from the notebooks dir or a SLURM job's CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, "hf"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_ROOT, "hf", "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_ROOT, "hf", "datasets"))
os.environ['PATH'] += ':/storage/home/hcoda1/9/kzhang430/.local/bin'

from datasets import load_dataset
from tqdm.auto import tqdm


CAND_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<a>[+\-−]?\d{1,3}(?:,\d{3})*|\d+)
    \s*(?P<op>[+\-*/×÷])\s*
    (?P<b>[+\-−]?\d{1,3}(?:,\d{3})*|\d+)
    \s*=\s*
    (?P<c>[+\-−]?\d{1,3}(?:,\d{3})*|\d+)
    (?![\w.])
    """,
    re.VERBOSE,
)

def _to_int(s: str) -> int:
    s = s.replace("−", "-").replace(",", "")
    return int(s)

def _apply(a: int, b: int, op: str):
    if op == "+": return a + b
    if op == "-": return a - b
    if op in ("*", "×"): return a * b
    if op in ("/", "÷"):
        if b == 0 or a % b != 0: return None
        return a // b
    return None

def extract_verified_expressions(text: str):
    out = []
    verified = 0
    candidates = 0

    for m in CAND_RE.finditer(text):
        candidates += 1
        a_s, b_s, c_s, op = m.group("a"), m.group("b"), m.group("c"), m.group("op")
        try:
            a, b, c = _to_int(a_s), _to_int(b_s), _to_int(c_s)
        except ValueError:
            continue

        val = _apply(a, b, op)
        if val is None or val != c:
            continue

        verified += 1
        out.append({"expr": f"{a_s} {op} {b_s} = {c_s}", "a": a, "b": b, "c": c, "op": op})

    rejected = candidates - verified
    return out, verified, rejected, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="finewine_arith_deduped.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--flush_every", type=int, default=2000)
    args = ap.parse_args()

    # This will DOWNLOAD + PREP the dataset the first time, then read locally after.
    
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
    total = len(ds) - args.start if hasattr(ds, "__len__") else None
    mode = "a" if args.append else "w"

    total_verified = total_rejected = total_candidates = 0
    docs_seen = 0

    with open(args.out, mode, encoding="utf-8", buffering=1) as f:
        for row_index, ex in tqdm(
            enumerate(islice(ds, args.start, None), start=args.start),
            total=total,
            desc="scan",
        ):
            docs_seen += 1
            if args.max_docs and docs_seen > args.max_docs:
                break

            text = ex.get("text", "")
            # robust in case something weird comes back
            if isinstance(text, (bytes, bytearray, memoryview)):
                text = bytes(text).decode("utf-8", errors="replace")
            else:
                text = str(text)

            matches, v, rj, cand = extract_verified_expressions(text)
            for mm in matches:
                rec = {"row_index": row_index, "expr": mm["expr"], "a": mm["a"], "op": mm["op"], "b": mm["b"], "c": mm["c"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total_verified += v
            total_rejected += rj
            total_candidates += cand

            if args.flush_every and (docs_seen % args.flush_every == 0):
                f.flush()
                os.fsync(f.fileno())

    print("done ->")
    print(f"Docs scanned: {docs_seen}")
    print(f"Total candidates: {total_candidates}")
    print(f"Total verified: {total_verified}")
    print(f"Total rejected: {total_rejected}")

if __name__ == "__main__":
    main()