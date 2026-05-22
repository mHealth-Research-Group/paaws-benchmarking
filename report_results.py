"""
Aggregate LOPO prediction CSVs into F1 scores and confusion matrix plots.

Outputs:
  {output_dir}/f1_scores.csv          — long-format per-participant + global F1
  {output_dir}/cm_{sensor}_{label_set}.png — 1-row confusion matrix grid
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from utils import MAPPING_SCHEMES

# Build inverse map: num_acts (int) -> mapping scheme name.
# All four schemes have unique output-class counts, so this is unambiguous.
_NUM_ACTS_TO_SCHEME = {len(set(v.values())): k for k, v in MAPPING_SCHEMES.items()}

_FILE_RE = re.compile(
    r"^(FL|SimFL_Lab)_(Left\w+|Right\w+)_(\d+)_Acts_(.+)_Participants_DS_(\d+)\.csv$"
)


def _plot_cm(ax, y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
    annots = np.array([
        [f"{cm_norm[i, j]:.2f}\n{cm[i, j]}" for j in range(len(classes))]
        for i in range(len(classes))
    ])
    sns.heatmap(
        cm_norm, annot=annots, fmt="", ax=ax,
        xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=1, cmap="Blues", cbar=False,
        square=True, linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)


def _parse_filename(fname):
    m = _FILE_RE.match(fname)
    if not m:
        return None
    num_acts = int(m.group(3))
    return {
        "protocol": m.group(1),
        "sensor": m.group(2),
        "label_set": _NUM_ACTS_TO_SCHEME.get(num_acts, f"{num_acts}_acts"),
        "dataset": m.group(4),
        "participant_id": int(m.group(5)),
    }


def main(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".csv"))

    # Per-participant F1 rows and grouped data for CMs.
    f1_rows = []
    groups = {}  # (sensor, label_set) -> {participants: [(id, df)], all_true, all_pred}

    for fname in files:
        meta = _parse_filename(fname)
        if meta is None:
            print(f"Warning: skipping unrecognised filename: {fname}")
            continue

        df = pd.read_csv(os.path.join(results_dir, fname), index_col=0)
        y_true = df["MAPPED_LABEL"]
        y_pred = df["PREDICTION"]

        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        f1_rows.append(
            {
                "label_set": meta["label_set"],
                "sensor": meta["sensor"],
                "participant_ID": meta["participant_id"],
                "wF1": wf1,
                "mF1": mf1,
            }
        )

        key = (meta["sensor"], meta["label_set"])
        if key not in groups:
            groups[key] = {"participants": [], "all_true": [], "all_pred": []}
        groups[key]["participants"].append((meta["participant_id"], df))
        groups[key]["all_true"].extend(y_true.tolist())
        groups[key]["all_pred"].extend(y_pred.tolist())

    part_df = pd.DataFrame(f1_rows)

    # Build global (ALL) rows: mean ± std formatted as string.
    global_rows = []
    for (sensor, label_set), data in groups.items():
        subset = part_df[(part_df["sensor"] == sensor) & (part_df["label_set"] == label_set)]
        wf1_mean, wf1_std = subset["wF1"].mean(), subset["wF1"].std()
        mf1_mean, mf1_std = subset["mF1"].mean(), subset["mF1"].std()
        global_rows.append(
            {
                "label_set": label_set,
                "sensor": sensor,
                "participant_ID": "ALL",
                "wF1": f"{wf1_mean:.3f} ± {wf1_std:.3f}",
                "mF1": f"{mf1_mean:.3f} ± {mf1_std:.3f}",
            }
        )

    final_df = pd.concat([part_df, pd.DataFrame(global_rows)], ignore_index=True)
    final_df = final_df[["label_set", "sensor", "participant_ID", "wF1", "mF1"]]

    csv_path = os.path.join(output_dir, "f1_scores.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Confusion matrix plots — one file per (sensor, label_set).
    for (sensor, label_set), data in groups.items():
        participants = sorted(data["participants"], key=lambda x: x[0])
        all_true = data["all_true"]
        all_pred = data["all_pred"]
        classes = sorted(set(all_true) | set(all_pred))

        n_cols = len(participants) + 1  # participants + global
        n_classes = len(classes)
        cell_size = 0.9
        subplot_w = max(3.5, cell_size * n_classes)
        fig_h = subplot_w + 1.5
        fig, axes = plt.subplots(1, n_cols, figsize=(subplot_w * n_cols, fig_h))
        if n_cols == 1:
            axes = [axes]

        for i, (pid, df) in enumerate(participants):
            _plot_cm(axes[i], df["MAPPED_LABEL"], df["PREDICTION"], classes, f"DS_{pid}")

        _plot_cm(axes[-1], all_true, all_pred, classes, "ALL")

        fig.suptitle(f"{sensor} — {label_set}", fontsize=13)
        plt.tight_layout()

        png_path = os.path.join(output_dir, f"cm_{sensor}_{label_set}.png")
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate LOPO results into F1 scores and confusion matrix plots."
    )
    parser.add_argument("--results_dir", default="replicated_results")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
