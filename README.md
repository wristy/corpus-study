# Investigating Human Alignment in LLM Arithmetic

Code and data for Investigating Human Alignment in LLM Arithmetic. It investigates the Pile (the Pythia training corpus) and explains the arithmetic behaviour of Pythia models — with the same effects that cognitive science has documented
in humans (e.g. the problem-size effect, and power-law-shaped number
production).

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
│   ├── 04b_arithmetic_2digit_model_sweep.ipynb  (Part-4 90x90 two-digit sweep, all sizes)
│   ├── 04c_sec_twodigit_model_sweep.ipynb       (Part-7 sweep = sec:twodigit, all sizes)
│   ├── 05_number_representation.ipynb    (latent number-line MDS)
│   └── 99_archive/                       (legacy notebooks kept for reference)
│
└── results/                  ← generated outputs
    ├── figures/               (PNGs, grouped flat — see "Result files" below)
    └── tables/
```

---

## Run-all instructions

All figures are written to `results/figures/`. Tables go to
`results/tables/`.

---

## Environment

See `environment.json`. 
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
| `figD_pythia*.png` | Supp: sec:twodigit (Part-7) model-size sweep, surprisal heatmaps per model |
| `part9b_inv_sweep_correlations.png` | Supp: same, accuracy/correlations vs. scale |
| `figD_2digit_pythia*.png` | Supp: Part-4 strictly-two-digit model-size sweep |
| `part9_2digit_accuracy_pythia-*.png`, `part9_2digit_surprisal_pythia-*.png` | Supp: same, split by metric |
| `part9_2digit_sweep_correlations.png` | Supp: two-digit accuracy/correlations vs. model size |
| `part5nb_number_mds_*.png`, `part5nb_number_line_*.png`, `part5nb_arith_*.png` | Latent number line / arithmetic representation |
| `fit_3a–3f_*.png` | Fig 3 by-operation variants |

Re-running the notebooks will overwrite these with freshly-computed versions.

---

## Model-size sweeps of the two-digit sections

Part 8 of `04_arithmetic_main.ipynb` sweeps the **single-digit** Part-5 analysis over every
Pythia size. Two further notebooks do the same for the two-digit sections. **They are not
interchangeable — pick the one matching the section you are extending:**

| Notebook | Replicates | Grid | Size metric | Frequency |
|---|---|---|---|---|
| `04c_sec_twodigit_model_sweep.ipynb` | **Part 7 — this is what `sec:twodigit` reports** | `a,b in 1..100`, 10,000 cells/op, inverse framing | `\|c\|` | raw Pile count |
| `04b_arithmetic_2digit_model_sweep.ipynb` | Part 4 (strictly two-digit) | `a,b in 10..99`, 8100 cells/op, inverse framing | `max(\|x\|,\|y\|,\|c\|)` | `log1p` Pile count |

### `04c` — the `sec:twodigit` sweep (use this one for the appendix)

Part 7 cell 83 of `04_arithmetic_main.ipynb` prints the paper's correlation sentence verbatim,
and `04c` reproduces it exactly at 12B (all eight correlations match the published values to
+/-0.000):

| | add | sub | mul | div |
|---|---|---|---|---|
| r(surprisal, size) | 0.316 | 0.373 | 0.605 | 0.440 |
| r(surprisal, freq) | -0.033 | -0.052 | -0.148 | -0.131 |

Outputs: `results/tables/part7_sweep_raw/<model>__<op>.csv` (raw, checkpointed),
`part9b_inv_stats_all_models.csv` / `.docx`, `part9b_inv_appendix.tex` (raw-frequency column,
matching the section) plus `_logfreq` and `_allmodels` variants, and
`results/figures/figD_<model>.png` — 2x2 surprisal heatmaps in the same style as the
single-digit appendix figures (accuracy is reported in the tables, not plotted).

> **Known discrepancy.** The accuracies reported in `sec:twodigit`
> (0.831 / 0.796 / 0.388 / 0.509 for add / sub / mul / div) do **not** match what Part 7's code
> produces (0.861 / 0.806 / 0.382 / **0.370**). That quadruple appears in no notebook output in
> this repo. The correlations in the same paragraph *do* match Part 7 exactly, so the accuracy
> figures appear to come from an earlier run than the correlations. Division is off by 14 points
> and needs resolving in the paper. `04c` has a verification cell that prints this comparison.

> **Grid note — Figure 4 of the paper is 1..99, this sweep is 1..100.**
> Part 7's code uses `range(1, 101)` while its markdown heading still says "operands 1-99".
> The paper's Figure 4 comes from the earlier **1..99** version: its tick labels are
> `1,20,40,60,80,99`, and its addition (0.628) and multiplication (3.185) means match the
> 1..99 grid exactly. On 1..100 those become 0.494 and 3.500. The cause is the **4-shot
> prefix**, not the grid endpoint: shuffling a 1..99 pool with seed 42 yields
> `2 + 71 = 73 | 34 + 75 = 109 | 15 + 75 = 90 | 47 + 13 = 60`, whereas a 1..100 pool yields
> `38 + 72 = 110 | 67 + 73 = 140 | ...` — entirely different demonstrations. Cropping 1..100
> results down to 1..99 therefore does *not* reproduce the figure (addition 0.495, not 0.628).
>
> This sweep deliberately uses **1..100**, matching the correlations reported in
> `sec:twodigit` (which it reproduces to +/-0.000), not Figure 4. If the figure is ever
> regenerated, note that it and the section's numbers currently come from different grids.

### `04b` — the strictly-two-digit (Part 4) sweep

A genuine `a,b in 10..99` sweep, kept as a secondary variant. It reproduces Part 4's 12B row
exactly (addition accuracy 0.8301, r_size 0.3362, r_logfreq -0.1938). Outputs use the
`twodigit_sweep_raw/`, `part9_2digit_*` and `figD_2digit_*` names, with the same
surprisal-only figure style.

### Both sweeps

Each is **checkpointed**: every `(model, operation)` result is cached as a CSV and skipped on
re-run, so the analysis/figure/LaTeX cells re-execute in seconds without reloading a model.
Delete the raw CSVs to force a recompute. With 8100-10,000 cells per operation nearly every
correlation is significant, so effect sizes carry the interpretation, not p-values; p-values
that underflow double precision are reported as `<1e-308`.