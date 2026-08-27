#!/usr/bin/env python3
"""
Scan the Pile for ALL standalone integers (not tied to arithmetic problems).

Mirrors scan_pile_local.py's setup, but instead of looking for verified
`a op b = c` expressions, it extracts every integer literal it sees and
aggregates the features needed to plot the distribution in the same style
as metrics.ipynb:

  * digit-count histogram   -> dict {n_digits: count}
  * log10(|x|+1) histogram  -> fixed bins (memory-safe, big-int safe)
  * exact value counts for small magnitudes (|x| <= SMALL_CAP)

The corpus is streamed (streaming=True) so we never materialize the full
~800GB Pile on disk. Aggregates are tiny and checkpointed to --out as JSON
every --flush_every docs, so partial results are usable if the job is killed.
"""
import os
import re
import math
import json
import argparse
from collections import Counter

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

# A standalone integer: optional sign, with or without thousands separators.
#   leading  (?<![\w.])  -> not glued to a word char or a '.' (skips the
#                           fractional/parts of decimals & version strings)
#   trailing (?![\w])    -> not glued to a word char
#            (?!\.\d)     -> not the integer part of a decimal (e.g. 3.14)
# Unlike scan_pile_local.py's CAND_RE we deliberately allow a trailing '.'
# that is NOT followed by a digit, so sentence-final integers ("...in 1999.")
# are kept rather than silently dropped.
NUM_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<n>[+\-−]?\d{1,3}(?:,\d{3})*|\d+)
    (?![\w])(?!\.\d)
    """,
    re.VERBOSE,
)

# Fixed log10 binning: [0, LOG_MAX) in LOG_BINS bins. 10^50 covers anything
# realistic; larger magnitudes are clamped into the last bin.
LOG_MAX = 50.0
LOG_BINS = 1000
LOG_BIN_W = LOG_MAX / LOG_BINS

# Keep exact per-value counts only for |x| <= SMALL_CAP (bounded memory).
SMALL_CAP = 10000


def _to_int(s: str) -> int:
    s = s.replace("−", "-").replace(",", "")
    return int(s)


def safe_log10_int(x: int) -> float:
    """log10(|x|+1) without unsafe float conversion of huge ints."""
    x = abs(int(x))
    n = x + 1
    if n == 1:
        return 0.0
    k = n.bit_length()
    shift = max(k - 53, 0)
    m = n >> shift
    log2n = math.log2(m) + shift
    return log2n / math.log2(10)


def make_state():
    return {
        "digit_counts": Counter(),          # n_digits -> count
        "log10_hist": [0] * LOG_BINS,       # binned log10(|x|+1)
        "log10_overflow": 0,                 # |x| >= 10^LOG_MAX
        "small_value_counts": Counter(),    # value -> count for |value| <= SMALL_CAP
        "total_numbers": 0,
        "docs_seen": 0,
    }


def update(state, text: str):
    dc = state["digit_counts"]
    lh = state["log10_hist"]
    sv = state["small_value_counts"]
    n_local = 0
    for m in NUM_RE.finditer(text):
        try:
            v = _to_int(m.group("n"))
        except ValueError:
            continue
        av = abs(v)
        # digit count
        nd = 1 if av == 0 else len(str(av))
        dc[nd] += 1
        # log10 bin
        lg = safe_log10_int(av)
        b = int(lg / LOG_BIN_W)
        if b >= LOG_BINS:
            state["log10_overflow"] += 1
        else:
            lh[b] += 1
        # small exact value counts (signed)
        if av <= SMALL_CAP:
            sv[v] += 1
        n_local += 1
    state["total_numbers"] += n_local


def serialize(state):
    return {
        "log_max": LOG_MAX,
        "log_bins": LOG_BINS,
        "log_bin_width": LOG_BIN_W,
        "small_cap": SMALL_CAP,
        "docs_seen": state["docs_seen"],
        "total_numbers": state["total_numbers"],
        "log10_overflow": state["log10_overflow"],
        "digit_counts": {str(k): int(v) for k, v in sorted(state["digit_counts"].items())},
        "log10_hist": [int(x) for x in state["log10_hist"]],
        "small_value_counts": {str(k): int(v) for k, v in sorted(state["small_value_counts"].items())},
    }


def write_out(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serialize(state), f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic, so a kill mid-write can't corrupt the file


def main():
    ap = argparse.ArgumentParser()
    # EleutherAI/the_pile_deduplicated is the deduplicated Pile (the corpus
    # the Pythia *-deduped models were trained on); streamable from the Hub.
    ap.add_argument("--dataset", default="EleutherAI/the_pile_deduplicated")
    ap.add_argument("--config", default=None, help="dataset config/name, if any")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--start", type=int, default=0, help="docs to skip")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="pile_number_distribution.json")
    ap.add_argument("--flush_every", type=int, default=20000)
    args = ap.parse_args()

    print(f"streaming {args.dataset} (config={args.config}, split={args.split})", flush=True)
    if args.config:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)

    if args.start:
        ds = ds.skip(args.start)

    state = make_state()

    pbar = tqdm(desc="scan", unit="doc")
    for ex in ds:
        if args.max_docs and state["docs_seen"] >= args.max_docs:
            break
        state["docs_seen"] += 1

        text = ex.get(args.text_key, "")
        if isinstance(text, (bytes, bytearray, memoryview)):
            text = bytes(text).decode("utf-8", errors="replace")
        else:
            text = str(text)

        update(state, text)

        pbar.update(1)
        if args.flush_every and (state["docs_seen"] % args.flush_every == 0):
            pbar.set_postfix(nums=state["total_numbers"])
            write_out(args.out, state)
    pbar.close()

    write_out(args.out, state)
    print("done ->", args.out, flush=True)
    print(f"Docs scanned:   {state['docs_seen']}")
    print(f"Total numbers:  {state['total_numbers']}")
    print(f"Distinct digit-lengths: {len(state['digit_counts'])}", flush=True)


if __name__ == "__main__":
    main()
    # The HF/pyarrow streaming backend spawns background threads that can
    # raise a fatal "PyGILState_Release" during normal interpreter shutdown
    # (after all output is already written). Hard-exit to skip that teardown
    # so the process returns a clean exit code 0.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
