#!/usr/bin/env python3
"""
make_figures.py -- publication figures for the multi-agent code complexity study.

Consumes the CSVs produced by extract_metrics.py and run_stats.py and renders
five figures as PNGs into analysis/figures/:

  fig1_distributions   box plots of all 5 metrics x 6 architectures x 2 models
  fig2_profiles        per-metric median profiles, both models overlaid (RQ2)
  fig3_effectsize      pairwise rank-biserial heatmap (post-hoc structure)
  fig4_accuracy        complexity vs pass@1 across the 12 cells
  fig5_rankdiagram     Friedman mean-rank diagram with significance grouping

Design: the six architectures fall into two complexity clusters (established by
run_stats.py); every figure colours by cluster so the partition is visible at a
glance. Figures 3 and 5 use SLOC under the primary (all-completions) condition
as the representative metric -- the post-hoc significance pattern is identical
across all five tested metrics, so one panel stands in for all (stated in caption).
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG_DIR = os.path.join(REPO, "analysis", "figures")

MODELS = ["gpt-4o", "gpt-4o-mini"]
ARCHS = ["Basic", "AC", "ACT", "Debugger", "AC+Debugger", "ACT+Debugger"]
ARCH_IDX = {a: i for i, a in enumerate(ARCHS)}
SHORT = {"Basic": "Basic", "AC": "AC", "ACT": "ACT", "Debugger": "Debugger",
         "AC+Debugger": "AC+Deb", "ACT+Debugger": "ACT+Deb"}

LEAN = {"Basic", "Debugger", "AC+Debugger"}
HEAVY = {"AC", "ACT", "ACT+Debugger"}
C_LEAN = "#0072B2"   # Okabe-Ito blue   -- colourblind-safe, distinct in greyscale
C_HEAVY = "#D55E00"  # Okabe-Ito vermillion
def cluster_colour(arch):
    return C_LEAN if arch in LEAN else C_HEAVY

TEST_METRICS = ["sloc", "cc", "halstead_volume", "halstead_difficulty", "halstead_effort"]
METRIC_LABEL = {"sloc": "SLOC", "cc": "Cyclomatic Complexity",
                "halstead_volume": "Halstead Volume", "halstead_difficulty": "Halstead Difficulty",
                "halstead_effort": "Halstead Effort"}
METRIC_SHORT = {"sloc": "SLOC", "cc": "CC", "halstead_volume": "Halstead V",
                "halstead_difficulty": "Halstead D", "halstead_effort": "Halstead E"}
LOG_METRICS = {"halstead_volume", "halstead_effort"}  # right-skewed, ratio-scale

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_metrics():
    """values[(model, arch, metric)] -> np.array; pass_rate[(model, arch)] -> float."""
    values = defaultdict(list)
    passed_seen = defaultdict(dict)   # (model,arch) -> {task: passed}
    with open(os.path.join(REPO, "analysis", "complexity_metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["value"] != "":
                values[(r["model"], r["architecture"], r["metric"])].append(float(r["value"]))
            passed_seen[(r["model"], r["architecture"])][r["task_id"]] = r["passed"] == "True"
    values = {k: np.array(v, float) for k, v in values.items()}
    pass_rate = {k: 100.0 * sum(d.values()) / len(d) for k, d in passed_seen.items()}
    return values, pass_rate


def load_csv(name):
    with open(os.path.join(REPO, "analysis", name)) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Figure 1 -- distributions
# --------------------------------------------------------------------------- #
def fig1_distributions(values):
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.8))
    for ri, model in enumerate(MODELS):
        for ci, metric in enumerate(TEST_METRICS):
            ax = axes[ri][ci]
            data = [values[(model, a, metric)] for a in ARCHS]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.3),
                            flierprops=dict(marker=".", markersize=3,
                                            markerfacecolor="0.4",
                                            markeredgecolor="none", alpha=0.5))
            for patch, a in zip(bp["boxes"], ARCHS):
                patch.set_facecolor(cluster_colour(a))
                patch.set_alpha(0.75)
                patch.set_edgecolor("0.25")
            if metric in LOG_METRICS:
                ax.set_yscale("log")
            ax.set_xticks(range(1, 7))
            ax.set_xticklabels([SHORT[a] for a in ARCHS], rotation=45, ha="right")
            if ri == 0:
                ax.set_title(METRIC_SHORT[metric])
            if ci == 0:
                ax.set_ylabel(f"{model}\n")
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.set_axisbelow(True)
    handles = [Patch(facecolor=C_LEAN, alpha=0.75, edgecolor="0.25", label="Lean cluster"),
               Patch(facecolor=C_HEAVY, alpha=0.75, edgecolor="0.25", label="Heavy cluster")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 -- per-metric median profiles, both models (RQ2 replication)
# --------------------------------------------------------------------------- #
def fig2_profiles(desc):
    med = {(r["model"], r["architecture"], r["metric"]): float(r["median"])
           for r in desc if r["condition"] == "primary"}
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))
    style = {"gpt-4o": dict(linestyle="-", marker="o"),
             "gpt-4o-mini": dict(linestyle="--", marker="s")}
    for ci, metric in enumerate(TEST_METRICS):
        ax = axes[ci]
        x = range(6)
        for model in MODELS:
            y = [med[(model, a, metric)] for a in ARCHS]
            ax.plot(x, y, color="0.35", linewidth=1.3, markersize=5,
                    zorder=2, **style[model])
            for xi, a in zip(x, ARCHS):
                ax.plot(xi, y[xi], marker=style[model]["marker"], markersize=5,
                        color=cluster_colour(a), markeredgecolor="0.25",
                        markeredgewidth=0.5, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SHORT[a] for a in ARCHS], rotation=45, ha="right")
        ax.set_title(METRIC_SHORT[metric])
        ax.set_ylabel("median")
        if metric in LOG_METRICS:
            ax.set_yscale("log")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color="0.35", **style["gpt-4o"], label="gpt-4o"),
               Line2D([0], [0], color="0.35", **style["gpt-4o-mini"], label="gpt-4o-mini"),
               Patch(facecolor=C_LEAN, label="Lean cluster"),
               Patch(facecolor=C_HEAVY, label="Heavy cluster")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 -- pairwise rank-biserial heatmap (post-hoc structure)
# --------------------------------------------------------------------------- #
def fig3_effectsize(posthoc, metric="sloc", condition="primary"):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6))
    for ax, model in zip(axes, MODELS):
        lut = {(r["arch_a"], r["arch_b"]): (float(r["rank_biserial"]),
               r["significant"] == "True")
               for r in posthoc if r["model"] == model
               and r["metric"] == metric and r["condition"] == condition}
        grid = np.full((6, 6), np.nan)
        for i in range(6):
            for j in range(i):
                rrb, sig = lut[(ARCHS[j], ARCHS[i])]   # a precedes b in ARCHS order
                grid[i][j] = rrb                        # -ve => row arch more complex
                txt = f"{rrb:+.2f}" + ("*" if sig else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if abs(rrb) > 0.55 else "black")
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels([SHORT[a] for a in ARCHS], rotation=45, ha="right")
        ax.set_yticklabels([SHORT[a] for a in ARCHS])
        for k, a in enumerate(ARCHS):
            ax.get_xticklabels()[k].set_color(cluster_colour(a))
            ax.get_yticklabels()[k].set_color(cluster_colour(a))
        ax.set_title(model)
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("matched-pairs rank-biserial r  (row vs column)\nnegative = row more complex")
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 -- complexity vs functional accuracy
# --------------------------------------------------------------------------- #
def fig4_accuracy(values, pass_rate, metric="sloc"):
    # One panel per model. Per-architecture label offsets separate the genuinely
    # near-coincident points (Debugger ~ AC+Debugger; AC ~ ACT+Debugger).
    lab = {
        "Basic":        ((0, -11), "center", "top"),
        "AC":           ((10, 0),  "left",   "center"),
        "ACT":          ((-8, 9),  "right",  "bottom"),
        "Debugger":     ((0, 10),  "center", "bottom"),
        "AC+Debugger":  ((0, -11), "center", "top"),
        "ACT+Debugger": ((8, -9),  "left",   "top"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.16, 3.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        for a in ARCHS:
            x = values[(model, a, metric)].mean()
            y = pass_rate[(model, a)]
            ax.scatter(x, y, s=75, color=cluster_colour(a), edgecolor="0.2",
                       linewidth=0.8, alpha=0.9, zorder=3)
            (dx, dy), ha, va = lab[a]
            ax.annotate(SHORT[a], (x, y), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=7.5, color="0.2")
        ax.set_title(model)
        ax.set_xlabel(f"mean {METRIC_SHORT[metric]} per cell  (complexity)")
        ax.grid(color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.22, y=0.22)
    axes[0].set_ylabel("pass@1  (%)")
    handles = [Patch(facecolor=C_LEAN, label="Lean cluster"),
               Patch(facecolor=C_HEAVY, label="Heavy cluster")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 -- Friedman mean-rank diagram with significance grouping
# --------------------------------------------------------------------------- #
def fig5_rankdiagram(meanranks, posthoc, metric="sloc", condition="primary"):
    # Critical-difference-style diagram: architectures sit on a rank axis;
    # leader lines fan out to stacked labels (lean cluster left, heavy right);
    # a thick bar joins each mutually non-significant group.
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.0))
    rows_y = [-0.44, -0.82, -1.20]
    tx_left, tx_right = -0.35, 6.35
    for ax, model in zip(axes, MODELS):
        mr = {r["architecture"]: float(r["mean_rank"]) for r in meanranks
              if r["model"] == model and r["metric"] == metric
              and r["condition"] == condition}
        nonsig = {frozenset((r["arch_a"], r["arch_b"])) for r in posthoc
                  if r["model"] == model and r["metric"] == metric
                  and r["condition"] == condition and r["significant"] != "True"}
        ax.set_xlim(-5.2, 11.5)
        ax.set_ylim(-1.55, 0.42)
        # rank axis
        ax.hlines(0, 1, 6, color="0.3", linewidth=1.3)
        for t in range(1, 7):
            ax.vlines(t, -0.045, 0.045, color="0.3", linewidth=1.0)
            ax.text(t, 0.22, str(t), ha="center", va="center", fontsize=7,
                    color="0.45")
        # significance-grouping bars (mutually non-significant clusters)
        for clus in (LEAN, HEAVY):
            if all(frozenset((x, y)) in nonsig
                   for x in clus for y in clus if x != y):
                xs = [mr[a] for a in clus]
                ax.plot([min(xs), max(xs)], [0.09, 0.09], color="0.2",
                        linewidth=4.5, solid_capstyle="round")
        # leader lines + stacked labels: lean cluster left, heavy right
        for clus, is_left in ((LEAN, True), (HEAVY, False)):
            ordered = sorted(clus, key=lambda a: mr[a])
            tx = tx_left if is_left else tx_right
            ha = "right" if is_left else "left"
            for row, a in enumerate(ordered):
                y = rows_y[row]
                ax.plot([mr[a], mr[a]], [0, y], color="0.6", linewidth=0.8)
                ax.plot([mr[a], tx], [y, y], color="0.6", linewidth=0.8)
                ax.scatter(mr[a], 0, s=70, color=cluster_colour(a),
                           edgecolor="0.2", linewidth=0.7, zorder=4)
                ax.text(tx + (-0.15 if is_left else 0.15), y,
                        f"{a}  ({mr[a]:.2f})", ha=ha, va="center", fontsize=7,
                        color=cluster_colour(a))
        ax.set_title(model, loc="left", fontsize=8, y=0.86)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(PNG_DIR, exist_ok=True)

    values, pass_rate = load_metrics()
    desc = load_csv("stats_descriptive.csv")
    posthoc = load_csv("stats_posthoc.csv")
    meanranks = load_csv("stats_meanranks.csv")

    figures = {
        "fig1_distributions": fig1_distributions(values),
        "fig2_profiles": fig2_profiles(desc),
        "fig3_effectsize": fig3_effectsize(posthoc),
        "fig4_accuracy": fig4_accuracy(values, pass_rate),
        "fig5_rankdiagram": fig5_rankdiagram(meanranks, posthoc),
    }
    for name, fig in figures.items():
        png = os.path.join(PNG_DIR, name + ".png")
        fig.savefig(png, dpi=200)
        plt.close(fig)
        print(f"  {name:22s} -> {os.path.relpath(png, REPO)}")
    print(f"\n{len(figures)} figures written to analysis/figures/.")


if __name__ == "__main__":
    main()
