import pandas as pd
from scipy.stats import wilcoxon

runs = pd.read_csv(r"c:\Users\NagaTharun\Downloads\Testing\Journal\results2\content\results\all_runs.csv")
res = pd.read_csv(r"c:\Users\NagaTharun\Downloads\Testing\Journal\results2\content\results\all_results.csv")


def pct(x):
    return f"{100 * x:.1f}"


for ds in ["SMS Spam", "AG News"]:
    print(f"\n=== {ds} classification ===")
    for _, r in res[res.Dataset == ds].iterrows():
        print(
            f" & {r['Model']} & {pct(r['Accuracy'])} & {pct(r['Precision'])} & "
            f"{pct(r['Recall'])} & {pct(r['F1'])} \\\\"
        )

for ds in ["SMS Spam", "AG News"]:
    print(f"\n=== {ds} efficiency ===")
    for _, r in res[res.Dataset == ds].iterrows():
        print(
            f" & {r['Model']} & {r['Train_s']:.1f} & {r['Infer_ms_per_1k']:.1f} & "
            f"{r['Size_MB']:.2f} & {r['Efficiency_E']:.2f} \\\\"
        )

for ds, a, b in [
    ("SMS Spam", "Linear SVM", "Random Forest"),
    ("AG News", "Logistic Regression", "Linear SVM"),
]:
    fa = runs[(runs.Dataset == ds) & (runs.Model == a)]["F1"].values
    fb = runs[(runs.Dataset == ds) & (runs.Model == b)]["F1"].values
    _, p = wilcoxon(fa, fb)
    print(f"Wilcoxon {ds}: {a} vs {b}, p={p}")

fastest = res.loc[res["Infer_ms_per_1k"].idxmin()]
print(f"\nFastest: {fastest['Model']} {fastest['Infer_ms_per_1k']:.3f} ms on {fastest['Dataset']}")
