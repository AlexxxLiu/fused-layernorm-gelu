"""Plot speedup and achieved bandwidth from results.csv."""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default="speedup.png")
    args = ap.parse_args()

    labels, speedups, gbps = [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            labels.append(f"{row['N']}x{row['H']}")
            speedups.append(float(row["speedup"]))
            gbps.append(float(row["fused_gbps"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.bar(labels, speedups, color="#3b7dd8")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity")
    ax1.set_ylabel("speedup vs eager")
    ax1.set_title("Fused LayerNorm+GELU vs PyTorch eager")
    ax1.legend()
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=8)

    ax2.bar(labels, gbps, color="#d8813b")
    ax2.set_ylabel("achieved GB/s (fused)")
    ax2.set_xlabel("shape (N x H)")
    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
