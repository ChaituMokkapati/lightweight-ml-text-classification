"""Regenerate fig2 and fig4 from all_results.csv (no retraining required)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results2" / "content" / "results" / "all_results.csv"
FIG_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

all_results = pd.read_csv(RESULTS_CSV)

# Figure 2: F1 vs Latency scatter
fig, ax = plt.subplots(figsize=(8, 6))
markers = {"SMS Spam": "o", "AG News": "s"}
for dataset, marker in markers.items():
    sub = all_results[all_results["Dataset"] == dataset]
    ax.scatter(sub["Infer_ms_per_1k"], sub["F1"], label=dataset, s=100, marker=marker)
    for _, r in sub.iterrows():
        ax.annotate(
            r["Model"],
            (r["Infer_ms_per_1k"], r["F1"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
ax.set_xlabel("Inference Latency (ms per 1,000 samples)")
ax.set_ylabel("F1-score")
ax.set_title("Accuracy-Efficiency Trade-off")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_f1_latency.png", dpi=200, bbox_inches="tight")
plt.close()

# Figure 4: Training time bar chart
models = list(all_results["Model"].unique())
x = np.arange(len(models))
width = 0.35
sms = all_results[all_results["Dataset"] == "SMS Spam"].set_index("Model").loc[models]["Train_s"]
ag = all_results[all_results["Dataset"] == "AG News"].set_index("Model").loc[models]["Train_s"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, sms, width, label="SMS Spam")
ax.bar(x + width / 2, ag, width, label="AG News")
ax.set_ylabel("Training Time (seconds)")
ax.set_title("Training Time by Model and Dataset")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_training_time.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved fig2 and fig4 to {FIG_DIR}")
