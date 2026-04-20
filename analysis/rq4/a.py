import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import ast

# 1. LOAD DATA
df = pd.read_csv(r"G:\663P\dataset\data\updated_dataset_2_metrics_f.csv")

# 2. HELPER FUNCTION FOR TUPLES
def extract_turnover_value(val):
    """
    Handles cases where turnover is a tuple (sha, value) or a string "(sha, value)".
    Returns the numeric value, or NaN if invalid/-1.
    """
    if pd.isna(val) or val == -1 or val == "-1":
        return np.nan
    
    try:
        # If it's a string representation of a tuple: "(sha, 0.5)"
        if isinstance(val, str) and "(" in val:
            parsed = ast.literal_eval(val)
            val = parsed[1]
        # If it's already a tuple/list
        elif isinstance(val, (tuple, list)):
            val = val[1]
        
        # Final conversion to float
        num = float(val)
        return num if num >= 0 else np.nan
    except (ValueError, SyntaxError, IndexError, TypeError):
        return np.nan

# 3. CLEANING & PROCESSING
# Define columns
turnover_cols = ["turnover_c5", "turnover_c10", "turnover_c20", "turnover_m1", "turnover_m3"]
doc_quality_cols = ["doc_entropy", "doc_code_overlap", "doc_redundancy"]

# Drop rows with no documentation baseline
df = df.dropna(subset=doc_quality_cols)
df = df[df['doc_lines'] > 0].copy()

# Process turnover columns (Extract numbers from tuples and filter -1)
print("Processing turnover columns...")
for col in turnover_cols:
    df[col] = df[col].apply(extract_turnover_value)

# Drop rows that have NaNs in the turnover columns after extraction
df_turn = df.dropna(subset=turnover_cols).copy()

# 4. SPLIT BY GROUP (Human vs Agent)
# Ensure case-insensitivity for the group label
df_h = df_turn[df_turn['group'].str.lower() == 'human'].copy()
df_a = df_turn[df_turn['group'].str.lower() == 'agent'].copy()

print(f"\n--- Data Summary ---")
print(f"Total Rows:  {len(df_turn)}")
print(f"Human Rows:  {len(df_h)}")
print(f"Agent Rows:  {len(df_a)}")

# 5. STATISTICAL ANALYSIS (Mann-Whitney U)
print("\n--- Statistical Comparison (Human vs Agent) ---")
for col in turnover_cols:
    group_h = df_h[col]
    group_a = df_a[col]
    
    if not group_h.empty and not group_a.empty:
        stat, p = mannwhitneyu(group_h, group_a, alternative='two-sided')
        
        print(f"\nMetric: {col}")
        print(f"  Human Median: {group_h.median():.4f} (Mean: {group_h.mean():.4f})")
        print(f"  Agent Median: {group_a.median():.4f} (Mean: {group_a.mean():.4f})")
        rbc = 1 - (2 * stat) / (len(group_h) * len(group_a))
        print(f"  rank-biserial correlation: {rbc:.4f}")
        print(f"  P-value:      {p:.6e}")
    else:
        print(f"\nMetric: {col} - Missing data for one or both groups.")

# 6. VISUALIZATION
# Reshape for Seaborn lineplot
df_melt = df_turn.melt(
    id_vars=["group"],
    value_vars=turnover_cols,
    var_name="time_interval",
    value_name="turnover_value"
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_melt, 
    x="time_interval", 
    y="turnover_value", 
    hue="group", 
    marker="o", 
    estimator="mean", 
    errorbar="se"
)

plt.title("Documentation Turnover: Human vs Agentic AI", fontsize=14)
plt.ylabel("Mean Turnover Rate (with Standard Error)", fontsize=12)
plt.xlabel("Time Horizon (Commits/Months)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()