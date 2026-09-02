#!/bin/bash
#SBATCH --job-name=agentdoc-mine

#SBATCH --time=35:00:00    
#SBATCH --cpus-per-task=3             # matches MAX_WORKERS = number of GITHUB_TOKEN_* set
#SBATCH --mem=16G
#SBATCH --output=logs/mine-%j.out
#SBATCH --error=logs/mine-%j.err
#
# Submit from the repo root on the login node with: sbatch hpc/run_build.sh
# Run hpc/setup_env.sh once beforehand to create the venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11

source "$REPO_ROOT/venv/bin/activate"

if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "ERROR: $REPO_ROOT/.env not found (needs GITHUB_TOKEN_1 etc.)" >&2
    exit 1
fi

# build.py loads .env itself via python-dotenv, but repo clones happen relative
# to CWD, so we run it from the repo root (matches OUTPUT_DIR = ./dataset/data).
python dataset/buildDataset/build.py
