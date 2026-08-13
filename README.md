# On the Naturalness of Agent-Generated Documentation

Replication package for the paper *On the Naturalness of Agent-Generated Documentation*. It contains the mining scripts used to build the corpus, the corpus itself, and one analysis notebook per research question.

## Repository layout

```
dataset/
  buildDataset/    mining scripts and notebooks
  data/            PR pools, raw mining output, and the final corpus
analysis/
  rq1/ … rq5/      one folder per research question
```

## Setup

**Python packages**

- Mining: `pandas`, `numpy`, `requests`, `python-dotenv`, `python-dateutil`, `tqdm`, `textstat`, `fastparquet`
- Analysis: `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `pingouin`, `scikit-learn`, `nltk`, `vaderSentiment`, `lexicalrichness`, `textstat`

**GitHub tokens**

Create a `.env` file in the repository root:

```
GITHUB_TOKEN_1=ghp_...
GITHUB_TOKEN_2=ghp_...
GITHUB_TOKEN_3=ghp_...
```

## Dataset

### Mining scripts - `dataset/buildDataset/`

- **`build.py`** - the primary miner. Takes a Parquet list of PRs, clones each repository at the merge commit, extracts functions and their documentation for the supported languages, and computes entropy, readability, complexity, and static-analysis metrics. 
- **`fetch_new_agent_prs.ipynb`** - searches GitHub for recently merged PRs authored by AI agents (Claude Code, Copilot, Cursor, Devin, OpenAI Codex), saving the PR list to `new_agent_pr_list.parquet` and the mined output to `agent_supplement_dataset.csv`.
- **`build_human_pr_list.ipynb`** - builds the human baseline PR pool with the query `is:pr is:merged created:<2021-01-01 stars:>=25`, saved to `human_baseline_2021.parquet`. 
- **`combine.ipynb`** - merges the agent and human mining output, folds in the supplementary agent PRs, re-tokenizes and cleans comments, and writes the final `dev_agent_combined.csv`.

### Data files - `dataset/data/`

**Final corpus**

- `dev_agent_combined.csv` - the cleaned dataset every analysis notebook loads.

**Raw mining output** (input to `combine.ipynb`)

- `agent_dataset_subset_data.csv` + `stats_agent_subset_data.json` - agent PRs, from `build.py`.
- `dev_dataset_subset_data.csv` + `stats_dev_subset_data.json` - human PRs, from `build.py`.
- `agent_supplement_dataset.csv` + `mining_stats_new_agent.json` - supplementary agent PRs, from `fetch_new_agent_prs.ipynb`.

**PR pools** (input to the miners)

- `all_pull_request.parquet` - the AIDev dataset, downloaded from HuggingFace rather than produced by anything in this repo.
- `all_pull_request_ballanced.parquet` - our sample of AIDev, 200 PRs per agent.
- `new_agent_pr_list.parquet` - the supplementary agent PRs found by `fetch_new_agent_prs.ipynb`.
- `human_baseline_2021.parquet` - the human baseline PR pool.

### Reading `dev_agent_combined.csv`

- **One row per function**, any PR-level count has to deduplicate on `(repo, pull_request)` first.
- **`label` identifies the agent** - `Claude_Code`, `Copilot`, `Cursor`, `Devin`, `OpenAI_Codex`. It is **null for human-authored rows**; that null is how the human group is selected.
- Per-function metrics: `loc`, `sloc`, `cyclomatic_complexity`, `num_parameters`, `doc_lines`, `doc_text`, `doc_entropy`, `total_entropy`, `doc_readability`, `doc_code_overlap`, `doc_redundancy`, `semgrep_findings`, `semgrep_findings_count`.
- PR-level fields repeated on every row: `pr_date_created`, `pr_date_merged`, `pr_date_closed`, `turnover_c5`.

## Analysis

Each RQ folder holds an `a.ipynb` that loads `../../dataset/data/dev_agent_combined.csv`. Run the notebooks top to bottom.

- `rq1/` - per-language comparison of entropy, code overlap, and token count. This directory also has a file which randomly selects 200 samples of documentation which was used for a qualitative analysis.
- `rq2/` - documented-vs-undocumented and human-vs-agent comparisons on code complexity metrics (cyclomatic complexity, SLOC, static analysis findings).
- `rq3/` - interface complexity evaluation.
- `rq4/` - repository activity and commit turnover comparisons.
- `rq5/` - sentiment and readability analysis of comments, and naturalness modeling: `n-gram.ipynb` (n-gram cross-entropy) and `rq5_robustness.ipynb`.

## Reproducing the dataset

Mining takes many hours and is rate-limited by GitHub. `dataset/data/` already contains every artifact below, so this is only needed to rebuild the corpus from scratch.

1. **Human PR pool.** Run `build_human_pr_list.ipynb` → `human_baseline_2021.parquet`.
2. **Supplementary agent PRs.** Run `fetch_new_agent_prs.ipynb` → `new_agent_pr_list.parquet`, then the mining cells in the same notebook → `agent_supplement_dataset.csv`.
3. **Mine the human PRs.** Run `build.py` and point the df path to `human_baseline_2021.parquet` and build will produce `dev_dataset_subset_data.csv`.
4. **Mine the agent PRs.** Run `build.py` and point the df path to `all_pull_request_ballanced.parquet` and build will produce `agent_dataset_subset_data.csv`.
5. **Combine.** Run `combine.ipynb` over `dev_dataset_subset_data.csv`, `agent_dataset_subset_data.csv`, and `agent_supplement_dataset.csv` → `dev_agent_combined.csv`.

Switching `build.py` between steps 3 and 4 means editing three lines in the `__main__` block - the input `pd.read_parquet(...)` call, `output_path`, and `stats_path`. Each configuration already present on commented out lines. Note that `build.py` appends to its output file, so delete a partial CSV before re-running.


