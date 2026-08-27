#!/usr/bin/env python3
"""
Extract per-layer hidden-state vectors for arithmetic expressions from a Pythia model.

Used by notebooks/05_number_representation.ipynb Part 9 (problem-size effect in the
latent arithmetic representation).

For every problem "a op b = c" and every prompt template we run one forward pass and read
the residual stream at the LAST token, at every layer (0 = embedding output, 1..L = output
of transformer block i). The answer vectors for c are NOT extracted here: they are the
single-number vectors already produced by extract_number_reps.py, so Part 9 needs that
cache too and both must come from the same model + revision.

Problem grid. Both operands are single digits 1-9 in every case:

    a + b = c   all 81 pairs                    c in   2..18
    a - b = c   all 81 pairs, negatives kept    c in  -8..8
    a * b = c   all 81 pairs                    c in   1..81
    a / b = c   the 23 pairs where b divides a  c in   1..9

Subtraction keeps negative and zero answers rather than being restricted to a >= b, and
division is the plain operand grid filtered to exact quotients, so division has 23 problems
where the other operations have 81. Answers therefore run -8..81, and the vectors for
0 and the negatives come from extract_number_extras.py rather than the 1-100 cache.

Output: data/representation/arith_hidden_states_<model>.npz
  vecs      float16 (n_templates, n_problems, n_layers+1, hidden)
  ops       <U1    (n_problems,)
  a, b, c   int64  (n_problems,)
  templates <U...  (n_templates,)
"""
import argparse, json, os
import numpy as np

TEMPLATES = ["{a} {op} {b}", "{a} {op} {b} ="]
OPS = ["+", "-", "*", "/"]


def build_problems(hi=9):
    """(op, a, b, c) for the four operand grids, grouped by operation.

    Every operand is a single digit 1..hi. Subtraction keeps negative and zero answers;
    division keeps only the pairs that divide exactly, so it contributes fewer problems.
    """
    P = []
    for a in range(1, hi + 1):
        for b in range(1, hi + 1):
            P.append(("+", a, b, a + b))
    for a in range(1, hi + 1):
        for b in range(1, hi + 1):
            P.append(("-", a, b, a - b))          # negatives and zero kept
    for a in range(1, hi + 1):
        for b in range(1, hi + 1):
            P.append(("*", a, b, a * b))
    for a in range(1, hi + 1):
        for b in range(1, hi + 1):
            if a % b == 0:                        # exact quotients only
                P.append(("/", a, b, a // b))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-12b-deduped")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--hi", type=int, default=9, help="largest operand of the grid")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, "hf", "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT, "hf", "datasets"))
    out_dir = args.out_dir or os.path.join(ROOT, "data", "representation")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    P = build_problems(args.hi)
    tag = args.model.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} rev={args.revision} device={device}", flush=True)
    per_op = {op: sum(1 for q in P if q[0] == op) for op in OPS}
    print(f"problems={len(P)} {per_op}  templates={TEMPLATES}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers} hidden={hidden}", flush=True)

    # ---- token audit: the read-out position must mean the same thing for every problem,
    #      i.e. the last token is the second operand (expression form) or "=" (equation form).
    audit = []
    for t in TEMPLATES:
        for op, a, b, c in P:
            ids = tok.encode(t.format(a=a, op=op, b=b), add_special_tokens=False)
            toks = tok.convert_ids_to_tokens(ids)
            audit.append({"template": t, "op": op, "a": a, "b": b, "c": c,
                          "n_tokens": len(ids), "last_token": toks[-1],
                          "tokens": toks})
    with open(os.path.join(out_dir, f"arith_token_audit_{tag}.json"), "w") as f:
        json.dump(audit, f, indent=1)
    for t in TEMPLATES:
        rows = [r for r in audit if r["template"] == t]
        want = " =" if t.endswith("=") else None      # None -> expect the second operand
        ok = sum(1 for r in rows
                 if r["last_token"].replace("Ġ", " ")
                 == (want if want else f" {r['b']}"))
        print(f"  template {t!r}: {sorted(set(r['n_tokens'] for r in rows))} tokens; "
              f"{ok}/{len(rows)} read out at the intended position", flush=True)

    # ---- extract
    vecs = np.zeros((len(TEMPLATES), len(P), n_layers + 1, hidden), dtype=np.float16)
    with torch.inference_mode():
        for ti, t in enumerate(TEMPLATES):
            for i, (op, a, b, c) in enumerate(P):
                ids = tok.encode(t.format(a=a, op=op, b=b),
                                 add_special_tokens=False, return_tensors="pt")
                out = model(input_ids=ids.to(model.device), output_hidden_states=True)
                last = ids.shape[1] - 1
                for L in range(n_layers + 1):          # 0 = embeddings
                    vecs[ti, i, L] = out.hidden_states[L][0, last].to(torch.float16).cpu().numpy()
            print(f"  done template {t!r}", flush=True)

    out_npz = os.path.join(out_dir, f"arith_hidden_states_{tag}.npz")
    np.savez_compressed(
        out_npz, vecs=vecs,
        ops=np.array([p[0] for p in P]),
        a=np.array([p[1] for p in P]), b=np.array([p[2] for p in P]),
        c=np.array([p[3] for p in P]),
        templates=np.array(TEMPLATES),
    )
    print(f"saved {out_npz}  shape={vecs.shape} "
          f"({os.path.getsize(out_npz) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
