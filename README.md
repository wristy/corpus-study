# Investigating Human Alignment in LLM Arithmetic

Code and data for a corpus study asking whether the statistical structure of
the Pile (the Pythia training corpus) explains the arithmetic behaviour of
Pythia models — with the same effects that cognitive science has documented
in humans (e.g. the problem-size effect, and power-law-shaped number
production).

This repository reproduces every figure, table, and statistic reported in
the paper.

---

## Layout

```
corpus-study/
├── README.md               ← you are here
├── environment.json        ← python/torch/model/GPU versions
│
├── data/                    ← input data files (.jsonl, .csv, .json)
│   ├── representation/                                (token audits; see note below)
│   ├── pile_arith_matches_full_fresh_deduped.jsonl    (204,073 verified arith expressions)
│   ├── pile_number_distribution.json                  (4.78 B integers aggregated)
│   ├── pythia_freq_results*.csv                       (per-Pile cell frequencies)
│   ├── generations*.csv / .json                       (model generations)
│   └── arith_0to99_allops_8shot_logprob.csv           (Part 8 sweep raw)
│
├── scripts/                 ← standalone scan scripts + SLURM job files
│   ├── scan_pile_local.py        (extract verified arith expressions)
│   ├── scan_pile_local.sbatch
│   ├── extract_number_reps.py    (per-layer hidden states for 1–100)
│   ├── extract_arith_reps.py     (per-layer hidden states for arithmetic prompts)
│   ├── extract_number_extras.py
│   ├── scan_numbers.py           (extract integer counts)
│   ├── scan_numbers.sbatch
│   └── scan_pile.sub
│
├── notebooks/                ← canonical, run-all order
│   ├── 00_setup.ipynb
│   ├── 01_corpus_scan.ipynb
│   ├── 02_number_distribution.ipynb      (Fig 1, Fig 4)
│   ├── 03_metrics_powerlaw.ipynb         (Fig 2, Fig 3)
│   ├── 04_arithmetic_main.ipynb          (Fig 5, Fig 6, Sec 3.5, Sec 3.6, Supp)
│   ├── 05_number_representation.ipynb    (latent number-line MDS)
│   └── 99_archive/                       (legacy notebooks kept for reference)
│
└── results/                  ← generated outputs
    ├── figures/               (PNGs, grouped flat — see "Result files" below)
    └── tables/
```

---

## Note on `data/representation/`

The per-layer activation caches used by `05_number_representation.ipynb`
(`number_hidden_states_1_100_*.npz`, `number_hidden_states_extra_*.npz`,
`arith_hidden_states_*.npz`) are large binary files and are **not** checked
into this repository. They are cheap to regenerate:

- `scripts/extract_number_reps.py` produces the number-representation caches
  (~5 min on GPU, then a one-time ~15 min CPU pass for the MDS fit).
- `scripts/extract_arith_reps.py` produces the arithmetic-representation
  caches.

The token audits that accompany those caches (small JSON files recording
which prompt template/tokenization was used for each entry) are included.

---

## Report → notebook map

| Research question | Notebook |
|---|---|
| RQ1 (numbers 1–100 power-law) | `02_number_distribution.ipynb` |
| RQ2 (arith freq pooled / by op) | `03_metrics_powerlaw.ipynb` |
| RQ3 (Pythia number production, incl. 1–1000 range replication) | `02_number_distribution.ipynb` |
| RQ4 (10×10 and 100×100 grids, 1- and 2-digit, 4 ops, 4-shot / zero-shot / model-size sweep) | `04_arithmetic_main.ipynb` |
| RQ5 (subproblem difficulty, prominent operands) | `04_arithmetic_main.ipynb` |
| Latent number line (is the learned number representation compressed?) | `05_number_representation.ipynb` |

---

## Run-all instructions

> **All paths assume Jupyter was launched from the repo root.**

1. Open `notebooks/00_setup.ipynb` and run all cells. Confirms HF caches,
   model checkpoints, GPU availability.
2. (Optional, slow) Open `notebooks/01_corpus_scan.ipynb`. The cached
   outputs are already in `data/` — skip unless you want to re-extract from
   the Pile.
3. `notebooks/02_number_distribution.ipynb` → Fig 1, Fig 4.
4. `notebooks/03_metrics_powerlaw.ipynb` → Fig 2, Fig 3.
5. `notebooks/04_arithmetic_main.ipynb` → Fig 5, Fig 6, all tables,
   Sec 3.5 / 3.6 figures, and the smaller-Pythia sweep figures. This
   notebook takes longest, since it sweeps all 8 Pythia sizes.
6. `notebooks/05_number_representation.ipynb` → latent number-line MDS.
   First run calls `scripts/extract_number_reps.py` (needs a GPU, ~5 min);
   afterwards it reads the cached `.npz` files and runs on CPU in ~15 min
   (1-D MDS is multi-modal, so every solution reported is the
   lowest-stress one over 1000 restarts × 4 seeds).

All figures are written to `results/figures/`. Tables go to
`results/tables/`.

---

## Environment

See `environment.json`. Key facts:

| | |
|---|---|
| python | 3.9.12 |
| torch | 2.7.0+cu126 |
| transformers | 4.52.4 |
| accelerate | 1.10.1 |
| model | EleutherAI/pythia-12b-deduped (revision step143000) |
| dtype | torch.float16 |
| GPU | NVIDIA H200 (150 GB) |

Pile deduplicated: `EleutherAI/the_pile_deduplicated`.

---

## Result files

The PNGs checked in under `results/figures/` were produced by prior runs of
these notebooks (including the legacy notebooks in `notebooks/99_archive/`).
They map onto the paper roughly as follows:

| Filename pattern | Figure |
|---|---|
| `fig1_numbers_1_100*`, `numbers_1_100*` | Fig 1 (numbers 1–100) |
| `fit_1a…1d_*` | Fig 1 (digit-count / log-distribution variants) |
| `figure2_result_freq.png`, `fig2_arith_freq_pooled.png` | Fig 2 (raw + log) |
| `fit_1c_log(a+b)_density.png`, `fit_1d_log(c)_density.png` | Fig 2 distributions |
| `figure3_per_op.png`, `fig3_arith_freq_by_op.png` | Fig 3 (per-op curves) |
| `fit_2a_digits(c)_by_op.png`, `fit_2b_log(c)_density_by_op.png` | Fig 3 variants |
| `number_distribution*.png`, `number_loglog*.png`, `number_multi_combined*.png` | Fig 4 |
| `supp_fig_number_production_1_1000.png` | Supp: number production, 1–1000 |
| `single_digit_addition_heatmaps.png` | Fig 5 (single-digit) |
| `four_op_surprisal_heatmaps.png` | Fig 5 (4-op) |
| `four_op_surprisal_heatmaps_zeroshot.png`, `four_op_surprisal_heatmaps_inverse_zeroshot.png` | Supp (zero-shot ablation) |
| `part3_sweep_inverse_correlation.png`, `part3_2digit_inverse_correlation.png` | Sec 3.5 (inverse-pair correlations) |
| `single_digit_carry_heatmaps.png`, `composite_score_analysis.png` | Sec 3.5 (carry) |
| `part4_2digit_accuracy_heatmaps.png`, `part4_2digit_surprisal_heatmaps.png` | Fig 6 |
| `part5_1digit_direct_surprisal_heatmaps.png` | Fig 6 (single-digit) |
| `part6_both_digit_surprisal_heatmaps.png`, `part7_inverse_both_digit_surprisal_heatmaps.png` | Fig 6 (1- + 2-digit, inverse framing) |
| `part8_part5_sweep_correlations.png`, `part8_surprisal_pythia-*.png` | Supp: model-size sweep |
| `part5nb_number_mds_*.png`, `part5nb_number_line_*.png`, `part5nb_arith_*.png` | Latent number line / arithmetic representation |
| `fit_3a–3f_*.png` | Fig 3 by-operation variants |

Re-running the notebooks will overwrite these with freshly-computed versions.
