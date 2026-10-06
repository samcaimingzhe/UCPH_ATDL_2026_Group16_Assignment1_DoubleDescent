#!/usr/bin/env python3
"""Create Figure 1 and a single CSV from completed training checkpoints."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    rows = []
    missing = []
    for width in range(1, 65):
        for trial in range(5):
            path = RESULTS / f"width_{width:02d}_trial_{trial}.pt"
            if not path.exists():
                missing.append(path.name)
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if not saved.get("complete"):
                missing.append(path.name + " (incomplete)")
                continue
            for metric in saved["history"]:
                rows.append({"width": width, "trial": trial, **metric})

    if not rows:
        raise RuntimeError("no completed jobs found in results/")
    if missing:
        print(f"warning: {len(missing)} expected jobs are missing or incomplete")

    columns = ["width", "trial", "epoch", "train_loss", "train_error", "test_loss",
               "clean_test_error", "noisy_test_error"]
    with open(RESULTS / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["width"], row["epoch"])].append(row)

    widths = sorted({row["width"] for row in rows})
    final_epoch = max(row["epoch"] for row in rows)
    final_widths = [width for width in widths if (width, final_epoch) in grouped]
    train_final = [np.mean([r["train_error"] for r in grouped[(w, final_epoch)]]) for w in final_widths]
    test_final = [np.mean([r["noisy_test_error"] for r in grouped[(w, final_epoch)]]) for w in final_widths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(final_widths, test_final, color="blue", label="Test")
    axes[0].plot(final_widths, train_final, color="cornflowerblue", linestyle="--", label="Train")
    axes[0].set(xlabel="ResNet18 width parameter", ylabel="Test / Train Error")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)

    available_epochs = sorted({row["epoch"] for row in rows})
    colors = plt.cm.viridis_r(LogNorm(vmin=1, vmax=final_epoch)(available_epochs))
    for epoch, color in zip(available_epochs, colors):
        epoch_widths = [width for width in widths if (width, epoch) in grouped]
        values = [np.mean([r["noisy_test_error"] for r in grouped[(w, epoch)]]) for w in epoch_widths]
        axes[1].plot(epoch_widths, values, color=color, alpha=0.22, linewidth=0.8)
    axes[1].set(xlabel="ResNet18 width parameter", ylabel="Test Error")
    scalar = plt.cm.ScalarMappable(norm=LogNorm(vmin=1, vmax=final_epoch), cmap="viridis_r")
    colorbar = fig.colorbar(scalar, ax=axes[1])
    colorbar.set_label("Epochs")

    fig.suptitle("Figure 1 reproduction: CIFAR-10 with 15% label noise")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure1.png", dpi=220)
    plt.close(fig)
    print(f"wrote {RESULTS / 'summary.csv'}")
    print(f"wrote {RESULTS / 'figure1.png'}")


if __name__ == "__main__":
    main()

