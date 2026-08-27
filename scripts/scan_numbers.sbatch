#!/bin/bash
#SBATCH --job-name=num_scan
#SBATCH --account=gts-svarma39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=108:00:00
#SBATCH --qos=inferno
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SCRATCH="/storage/scratch1/9/${USER}"

# Activate the env by its prefix path (the named "pile-scan" env in $HOME got
# corrupted -- it had no bin/, so activation silently fell back to base python
# without `datasets`). This env lives on scratch and is built fresh.
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH}/envs/pile-scan"

# Stable cache location so the dataset stays downloaded between runs
CACHE_BASE="${SCRATCH}/hf_cache"
mkdir -p "$CACHE_BASE/hf" "$CACHE_BASE/hub" "$CACHE_BASE/datasets"

export HF_HOME="$CACHE_BASE/hf"
export HF_HUB_CACHE="$CACHE_BASE/hub"
export HF_DATASETS_CACHE="$CACHE_BASE/datasets"

OUT="${SCRATCH}/pile_number_distribution.json"

python -c "import datasets, tqdm; print('env OK')"

# --start lets you resume by skipping already-scanned docs.
python -u scan_numbers.py \
  --dataset EleutherAI/the_pile_deduplicated \
  --split train \
  --start 0 \
  --out "$OUT" \
  --flush_every 20000
