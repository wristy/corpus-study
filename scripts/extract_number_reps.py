#!/usr/bin/env python3
"""
Extract per-layer hidden-state vectors for the integers 1-100 from a Pythia model.

Used by notebooks/05_number_representation.ipynb (latent number-line MDS,
replicating Shah et al. 2023 Findings-ACL Fig 5 / Sec 4.4 at 1-100 scale).

For each integer n and each prompt template, we run one forward pass and read
the residual stream at the position of n's token, at every layer (0 = embedding
output, 1..L = output of transformer block i).

Output: data/representation/number_hidden_states_1_100_<model>.npz
  vecs      float16 (n_templates, 100, n_layers+1, hidden)
  numbers   int64   (100,)
  templates <U…    (n_templates,)
"""
import argparse, json, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-12b-deduped")
    ap.add_argument("--revision", default="step143000")
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

    NUMBERS = list(range(1, 101))
    # {} is replaced by the integer. The read-out position is the LAST token,
    # which the audit below verifies is exactly the token for the number.
    TEMPLATES = ["{}", " {}", "The number {}"]

    tag = args.model.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model={args.model} rev={args.revision} device={device}", flush=True)

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

    # ---- token audit: how many tokens does each number occupy in each template?
    audit = []
    for t in TEMPLATES:
        for n in NUMBERS:
            ids = tok.encode(t.format(n), add_special_tokens=False)
            toks = tok.convert_ids_to_tokens(ids)
            # tokens contributed by the number itself = full minus the prefix
            pre = tok.encode(t.format("").rstrip() if t.strip() != "{}" else "",
                             add_special_tokens=False)
            audit.append({"template": t, "n": n, "n_tokens_total": len(ids),
                          "n_tokens_prefix": len(pre),
                          "n_tokens_number": len(ids) - len(pre),
                          "last_token": toks[-1]})
    with open(os.path.join(out_dir, f"number_token_audit_{tag}.json"), "w") as f:
        json.dump(audit, f, indent=1)
    for t in TEMPLATES:
        nt = [a["n_tokens_number"] for a in audit if a["template"] == t]
        print(f"  template {t!r}: number occupies {sorted(set(nt))} token(s); "
              f"{sum(1 for x in nt if x == 1)}/100 single-token", flush=True)

    # ---- extract
    vecs = np.zeros((len(TEMPLATES), len(NUMBERS), n_layers + 1, hidden), dtype=np.float16)
    with torch.inference_mode():
        for ti, t in enumerate(TEMPLATES):
            for i, n in enumerate(NUMBERS):
                ids = tok.encode(t.format(n), add_special_tokens=False, return_tensors="pt")
                out = model(input_ids=ids.to(model.device), output_hidden_states=True)
                last = ids.shape[1] - 1
                for L in range(n_layers + 1):          # 0 = embeddings
                    vecs[ti, i, L] = out.hidden_states[L][0, last].to(torch.float16).cpu().numpy()
            print(f"  done template {t!r}", flush=True)

    out_npz = os.path.join(out_dir, f"number_hidden_states_1_100_{tag}.npz")
    np.savez_compressed(out_npz, vecs=vecs, numbers=np.array(NUMBERS),
                        templates=np.array(TEMPLATES))
    print(f"saved {out_npz}  shape={vecs.shape} "
          f"({os.path.getsize(out_npz)/1e6:.0f} MB)", flush=True)

if __name__ == "__main__":
    main()
