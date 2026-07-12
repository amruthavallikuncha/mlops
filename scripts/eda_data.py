import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/heart-risk-matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_risk.config import PLOT_DIR, PROCESSED_DATA_PATH, TARGET

sns.set_theme(style="whitegrid", context="notebook")


def save_current(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = pd.read_csv(PROCESSED_DATA_PATH)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    axes = data[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(
        bins=18, figsize=(12, 7), color="#2878B5", edgecolor="white"
    )
    for axis in axes.flat:
        axis.set_ylabel("Patient count")
    plt.suptitle("Distribution of continuous clinical measurements", y=1.02)
    save_current("feature_histograms.png")

    plt.figure(figsize=(11, 8))
    sns.heatmap(data.corr(numeric_only=True), cmap="vlag", center=0, square=False)
    plt.title("Feature correlation matrix")
    save_current("correlation_heatmap.png")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=data, x=TARGET, hue=TARGET, palette=["#4C956C", "#D1495B"], legend=False
    )
    ax.set(xlabel="Heart disease (0 = absent, 1 = present)", ylabel="Patient count")
    ax.set_title("Binary target distribution")
    save_current("class_distribution.png")

    missing = data.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    sns.barplot(x=missing.index, y=missing.values, color="#F4A261")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing values")
    plt.title("Missing value profile before pipeline imputation")
    save_current("missing_values.png")

    subset = data[["age", "thalach", "oldpeak", TARGET]].copy()
    pair = sns.pairplot(subset, hue=TARGET, corner=True, palette=["#4C956C", "#D1495B"])
    pair.fig.suptitle("Selected feature relationships by outcome", y=1.02)
    pair.savefig(PLOT_DIR / "feature_relationships.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    summary = {
        "records": int(len(data)),
        "features": int(data.shape[1] - 1),
        "positive_rate": round(float(data[TARGET].mean()), 4),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_by_column": {key: int(value) for key, value in missing.items()},
    }
    (PLOT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
