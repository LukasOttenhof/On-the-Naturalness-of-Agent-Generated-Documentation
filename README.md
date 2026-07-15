# On the Naturalness of Agent-Generated Documentation

This repository contains the replication package for the paper, On the Naturalness of Agent-Generated Documentation. This package includes dataset construction, corpus preparation, and exploratory analysis across multiple research questions.

## Project Overview

- `dataset/` contains dataset creation scripts, repository cloning helpers, and the CSV/Parquet data used to build the evaluation corpus.
- `analysis/` contains one folder per research question (`rq1`–`rq5`), each with the notebook(s) used to produce that RQ's statistics and figures.

## Key Components

### Dataset Construction

`dataset/buildDataset/` contains the scripts/notebooks used to mine and assemble the corpus:

- `build.py` — primary miner. Clones repositories, extracts documentation/code artifacts, and computes metrics. 
- `fetch_new_agent_prs.ipynb` — searches GitHub for recently merged PRs authored by AI agents (Claude Code, Copilot, Cursor, Devin, OpenAI Codex) and builds `new_agent_dataset.csv`.
- `build_human_pr_list.ipynb` — mines pre-2021 human PR baseline (`human_baseline_2021.parquet`) used as the human comparison group.
- `combine.ipynb` — post-processes the mined data (re-tokenizes/cleans comments, recomputes readability metrics) to produce a cleaned dataset used by some of the analysis notebooks. It combines the agent and dev only dataset created by build.py, and can also combine in the supplementary agent prs.

The dataset creation process requires GitHub API tokens configured in a `.env` file (`GITHUB_TOKEN_1`, `GITHUB_TOKEN_2`, `GITHUB_TOKEN_3`).

Key files in `dataset/data/`:

- `final_dataset_new_corrected.csv` — the primary, cleaned dataset used by notebooks.
- `final_dataset_new.csv` — the dataset prior to clean (input to `combine.ipynb`).
- `agent_final_dataset_subset_b.csv` / `dev_final_dataset_subset_b.csv` — raw per-agent / per-human mining output from `build.py`.
- `new_agent_dataset.csv`, `new_agent_pr_list.parquet`, `human_baseline_2021.parquet`, `all_pull_request.parquet`, `all_pull_request_ballanced.parquet` — intermediate PR pools used while building the corpus.
- `all_pull_request.parquet` — aidev dataset, `all_pull_request_ballanced.parquet`, is the subset or prs made by us
### Analysis

Each RQ folder under `analysis/` holds an `a.ipynb` notebook that loads `final_dataset_new_corrected.csv`and produces that question's statistics/plots:

- `rq1/` — per-language comparison of entropy, code overlap, and token count.
- `rq2/` — documented-vs-undocumented and human-vs-agent comparisons on code complexity metrics (cyclomatic complexity, SLOC, static analysis findings).
- `rq3/` — interface complexity evaluation.
- `rq4/` — repository activity and commit turnover comparisons.
- `rq5/` — sentiment and readability analysis of comments, and naturalness modeling: `n-gram.ipynb` (n-gram cross-entropy) and `rq5_robustness.ipynb`.

## Usage

- Open and run the notebooks under `analysis/rq1`–`analysis/rq5` to reproduce each research question's experiments and visualizations.

