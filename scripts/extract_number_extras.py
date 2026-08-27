#!/usr/bin/env python3
"""
Extract per-layer hidden-state vectors for integers OUTSIDE the 1-100 range covered by
extract_number_reps.py.

Used by notebooks/05_number_representation.ipynb Part 10, which scores each answer c against
its neighbours c-1 and c+1. Problems with c = 1 need a vector for 0, which the main number
cache does not hold. (No vector above 100 is needed: the largest answer in the grid is 81, so
the largest foil is 82.)

Same templates, same read-out position and same dtype as extract_number_reps.py, so the
vectors are interchangeable with those in the main cache.

Output: data/representation/number_hidden_states_extra_<model>.npz
  vecs      float16 (n_templates, n_numbers, n_layers+1, hidden)
  numbers   int64   (n_numbers,)
  templates <U...   (n_templates,)
"""
import argparse, json, os
import numpy as np

TEMPLATES = ["{}", " {}", "The number {}"]      # must match extract_number_reps.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-12b-deduped")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--numbers", default="0", help="comma-separated integers")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, "hf", "hub"))
    out_dir = args.out_dir or os.path.join(ROOT, "data", "representation")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    NUMBERS = [int(x) for x in args.numbers.split(",")]
    tag = args.model.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} rev={args.revision} device={device} numbers={NUMBERS}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    print(f"  n_layers={n_layers} hidden={hidden}", flush=True)

    audit = []
    for t in TEMPLATES:
        for n in NUMBERS:
            ids = tok.encode(t.format(n), add_special_tokens=False)
            toks = tok.convert_ids_to_tokens(ids)
            pre = tok.encode(t.format("").rstrip() if t.strip() != "{}" else "",
                             add_special_tokens=False)
            audit.append({"template": t, "n": n, "n_tokens_total": len(ids),
                          "n_tokens_prefix": len(pre),
                          "n_tokens_number": len(ids) - len(pre), "last_token": toks[-1]})
            print(f"  {t.format(n)!r} -> {toks}", flush=True)
    with open(os.path.join(out_dir, f"number_token_audit_extra_{tag}.json"), "w") as f:
        json.dump(audit, f, indent=1)

    vecs = np.zeros((len(TEMPLATES), len(NUMBERS), n_layers + 1, hidden), dtype=np.float16)
    with torch.inference_mode():
        for ti, t in enumerate(TEMPLATES):
            for i, n in enumerate(NUMBERS):
                ids = tok.encode(t.format(n), add_special_tokens=False, return_tensors="pt")
                out = model(input_ids=ids.to(model.device), output_hidden_states=True)
                last = ids.shape[1] - 1
                for L in range(n_layers + 1):
                    vecs[ti, i, L] = out.hidden_states[L][0, last].to(torch.float16).cpu().numpy()

    out_npz = os.path.join(out_dir, f"number_hidden_states_extra_{tag}.npz")
    np.savez_compressed(out_npz, vecs=vecs, numbers=np.array(NUMBERS),
                        templates=np.array(TEMPLATES))
    print(f"saved {out_npz}  shape={vecs.shape}", flush=True)


if __name__ == "__main__":
    main()
