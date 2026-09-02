#!/bin/bash
# One-time environment setup for the Digital Research Alliance of Canada (Compute Canada) clusters.
# Run this ONCE on a login node after cloning the repo and switching to this branch:
#
#   bash hpc/setup_env.sh
#
# It creates a venv under the repo root and installs everything build.py needs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

module purge
module load StdEnv/2023 python/3.11 git/2.42.0

python -m venv "$REPO_ROOT/venv"
source "$REPO_ROOT/venv/bin/activate"

pip install --upgrade pip

# Mining deps (see README.md "Setup") + tree_sitter deps build.py imports directly.
pip install \
    pandas numpy requests python-dotenv python-dateutil tqdm textstat fastparquet \
    semgrep lizard tree_sitter tree_sitter_language_pack

deactivate

echo ""
echo "Setup complete. venv created at $REPO_ROOT/venv"
echo "Make sure $REPO_ROOT/.env exists with GITHUB_TOKEN_1 (and optionally _2, _3) set,"
echo "then submit the mining job with: sbatch hpc/run_build.sh"
