import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


PUBLICATION_RC = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.8,
    "axes.linewidth":    1.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.dpi":        100,
}

plt.rcParams.update(PUBLICATION_RC)


def get_prediction_set_counts(qhat_csv, test_csv, alpha_target=0.1, verbose=True):
    qhat_df = pd.read_csv(qhat_csv)
    test_df = pd.read_csv(test_csv)

    row_idx = (qhat_df["alpha"] - alpha_target).abs().idxmin()
    row = qhat_df.loc[row_idx]

    alpha_used = float(row["alpha"])
    qhat_bll = float(row["qhat_BLL"])
    qhat_fsrq = float(row["qhat_FSRQ"])

    score_bll = 1.0 - test_df["BLL"].to_numpy()
    score_fsrq = 1.0 - test_df["FSRQ"].to_numpy()

    include_bll = score_bll <= qhat_bll
    include_fsrq = score_fsrq <= qhat_fsrq

    bll_only = int(np.sum(include_bll & ~include_fsrq))
    fsrq_only = int(np.sum(~include_bll & include_fsrq))
    both_sets = int(np.sum(include_bll & include_fsrq))
    empty_sets = int(np.sum(~include_bll & ~include_fsrq))
    total = bll_only + fsrq_only + both_sets + empty_sets

    results = {
        "alpha_requested": float(alpha_target),
        "alpha_used": alpha_used,
        "qhat_BLL": qhat_bll,
        "qhat_FSRQ": qhat_fsrq,
        "BLL_only": bll_only,
        "FSRQ_only": fsrq_only,
        "Both": both_sets,
        "Empty": empty_sets,
        "Total": total,
        "include_bll": include_bll,
        "include_fsrq": include_fsrq,
        "test_df": test_df,
    }

    if verbose:
        print(f"Requested alpha : {alpha_target}")
        print(f"Using alpha     : {alpha_used}")
        print(f"qhat_BLL        : {qhat_bll}")
        print(f"qhat_FSRQ       : {qhat_fsrq}")

        print("\nPrediction-set counts at fixed alpha")
        print("BLL only :", bll_only)
        print("FSRQ only:", fsrq_only)
        print("Both set :", both_sets)
        print("Empty set:", empty_sets)
        print("Total    :", total)

    return results


def plot_prediction_set_distribution(
    results,
    save_path=None,
    show=True,
    figsize=(6.0, 5.0),
    title=None,
):
    categories = ["BLL only", "FSRQ only", "Both", "Empty"]
    counts = [
        results["BLL_only"],
        results["FSRQ_only"],
        results["Both"],
        results["Empty"],
    ]
    colors = ["red", "blue", "green", "gray"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, counts, color=colors, alpha=0.82, edgecolor="black", linewidth=0.7)

    if title is None:
        title = rf"Prediction Set Distribution $\alpha={results['alpha_used']:.2f}$"

    ax.set_title(title)
    ax.set_xlabel("Prediction Set Category")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    ymax = max(counts) if len(counts) else 1
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(1, ymax) * 0.025,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def analyze_prediction_sets(
    qhat_csv,
    test_csv,
    alpha_target=0.1,
    save_path=None,
    show=True,
    verbose=True,
    title=None,
):
    results = get_prediction_set_counts(
        qhat_csv=qhat_csv,
        test_csv=test_csv,
        alpha_target=alpha_target,
        verbose=verbose,
    )

    plot_prediction_set_distribution(
        results,
        save_path=save_path,
        show=show,
        title=title,
    )

    return results


def plot_histograms(bin_centers, hist, label, title, save_path=None, figsize=(6.0, 5.0)):
    bar_width = bin_centers[1] - bin_centers[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        bin_centers,
        hist,
        width=bar_width,
        alpha=0.75,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
        label=label,
    )

    ax.set_title(title)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_and_plot_histograms_test_csv(
    test_csv_path,
    title,
    save_path=None,
    figsize=(6.0, 5.0),
    no_of_bins=20,
):
    sns.set_theme(context="paper", style="whitegrid", rc=PUBLICATION_RC)

    test_df = pd.read_csv(test_csv_path)
    predictions = test_df["FSRQ"].to_numpy()

    bll_count = int(np.sum(predictions < 0.5))
    fsrq_count = int(np.sum(predictions >= 0.5))

    print("BLL count :", bll_count)
    print("FSRQ count:", fsrq_count)

    hist_predictions, bin_edges_predictions = np.histogram(
        predictions, bins=no_of_bins, range=(0, 1)
    )

    bar_width = bin_edges_predictions[1] - bin_edges_predictions[0]
    bin_centers = bin_edges_predictions[:-1] + bar_width / 2

    fig, ax = plt.subplots(figsize=figsize)

    bll_handle = None
    fsrq_handle = None

    for center, count in zip(bin_centers, hist_predictions):
        if center < 0.5:
            bar = ax.bar(
                center,
                count,
                width=bar_width,
                color="red",
                alpha=0.72,
                edgecolor="black",
                linewidth=0.5,
                label="BLL",
            )
            if bll_handle is None:
                bll_handle = bar
        else:
            bar = ax.bar(
                center,
                count,
                width=bar_width,
                color="blue",
                alpha=0.72,
                edgecolor="black",
                linewidth=0.5,
                label="FSRQ",
            )
            if fsrq_handle is None:
                fsrq_handle = bar

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("FSRQ Probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    legend_handles = []
    legend_labels = []

    if bll_handle is not None:
        legend_handles.append(bll_handle[0])
        legend_labels.append("BLL")
    if fsrq_handle is not None:
        legend_handles.append(fsrq_handle[0])
        legend_labels.append("FSRQ")

    ax.legend(legend_handles, legend_labels, loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()