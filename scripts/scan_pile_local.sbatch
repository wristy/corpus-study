#!/bin/bash
#SBATCH --job-name=pile_scan
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

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pile-scan

# Stable cache location so the dataset stays downloaded
SCRATCH="/storage/scratch1/9/${USER}"
CACHE_BASE="${SCRATCH}/hf_cache"
mkdir -p "$CACHE_BASE/hf" "$CACHE_BASE/hub" "$CACHE_BASE/datasets"

export HF_HOME="$CACHE_BASE/hf"
export HF_HUB_CACHE="$CACHE_BASE/hub"
export HF_DATASETS_CACHE="$CACHE_BASE/datasets"

OUT="${SCRATCH}/pile_deduped_arith_matches_full_fresh_deduped.jsonl"

python -u scan_pile_local.py --start 0 --out "$OUT" --flush_every 2000