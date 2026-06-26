#!/usr/bin/env python3
"""
run_stats.py -- paired non-parametric statistical pipeline for the multi-agent code complexity study.

Consumes analysis/complexity_metrics.csv (produced by extract_metrics.py) and runs
the locked statistical recipe -- a paired-design adaptation of Della Porta et al.'s
independent-samples pipeline:

  * Omnibus: Friedman's test, one per (model, metric, correctness condition).
    2 models x 5 metrics x 2 conditions = 20 omnibus tests (10 primary, 10 secondary).
  * Effect size (omnibus): Kendall's W = Q / (n (k-1)), k = 6.
  * Post-hoc (only where omnibus rejects at alpha = 0.05): Wilcoxon signed-rank for
    all C(6,2) = 15 architecture pairs, two-sided, with Holm step-down correction
    within each 15-pair family.
  * Effect size (pairwise): matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-).

Conditions:
  * primary  (all-completions): every parse-valid completion with the expected
    entry point, regardless of test outcome.
  * passing  (passing-only, secondary robustness): tasks on which all six
    architectures (same model) produce a valid completion that passes the tests.

Missing data: listwise deletion at the task level within each (model, metric,
condition) family -- a task is dropped if any of the six architectures lacks a
usable value (and, for `passing`, if any fails the reference tests). Friedman
requires complete blocks.

Outputs (CSV, under analysis/):
  stats_omnibus.csv      one row per (model, metric, condition)
  stats_posthoc.csv      one row per pair, for significant omnibus families
  stats_meanranks.csv    Friedman mean rank per architecture (feeds rank diagram)
  stats_descriptive.csv  per-(model, architecture, condition, metric) summary
"""
import csv
import os
import warnings

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(REPO, "analysis", "complexity_metrics.csv")


def out_path(name):
    return os.path.join(REPO, "analysis", name)


MODELS = ["gpt-4o", "gpt-4o-mini"]
ARCHS = ["Basic", "AC", "ACT", "Debugger", "AC+Debugger", "ACT+Debugger"]
ARCH_IDX = {a: i for i, a in enumerate(ARCHS)}

# Formally hypothesis-tested metrics (m in {SLOC, CC, V, D, E}).
TEST_METRICS = ["sloc", "cc", "halstead_volume", "halstead_difficulty", "halstead_effort"]
# All metrics carried for descriptive reporting.
DESC_METRICS = ["loc", "sloc", "comments", "multi", "blank", "cc",
                "halstead_volume", "halstead_difficulty", "halstead_effort"]

ALPHA = 0.05
K = 6
DF = K - 1
MIN_N = 15  # below this n, omit the inferential test and flag the cell

# The 15 pairwise comparisons in the paper's canonical order, each tagged with
# the layer(s) that change and the mechanism class (Table I of the methodology).
# Causal-attribution claims are restricted to Single pairs.
PAIRS = [
    ("Basic", "AC",                 "R",       "Single"),
    ("Basic", "ACT",                "R,T",     "Compound"),
    ("Basic", "Debugger",           "D",       "Single"),
    ("Basic", "AC+Debugger",        "R,D",     "Compound"),
    ("Basic", "ACT+Debugger",       "R,T,D",   "Compound"),
    ("AC", "ACT",                   "T",       "Single"),
    ("AC", "Debugger",              "R<->D",   "Swap"),
    ("AC", "AC+Debugger",           "D",       "Single"),
    ("AC", "ACT+Debugger",          "T,D",     "Compound"),
    ("ACT", "Debugger",             "R,T,D",   "Compound"),
    ("ACT", "AC+Debugger",          "T<->D",   "Swap"),
    ("ACT", "ACT+Debugger",         "D",       "Single"),
    ("Debugger", "AC+Debugger",     "R",       "Single"),
    ("Debugger", "ACT+Debugger",    "R,T",     "Compound"),
    ("AC+Debugger", "ACT+Debugger", "T",       "Single"),
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data():
    """Return data[(model, arch)][task_id] = {'m': {metric: float|None},
    'passed': bool, 'valid': bool}."""
    data = {}
    with open(IN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["architecture"])
            cell = data.setdefault(key, {}).setdefault(
                row["task_id"],
                {"m": {}, "passed": row["passed"] == "True",
                 "valid": (row["parse_valid"] == "True"
                           and row["entry_point_present"] == "True")},
            )
            val = row["value"]
            cell["m"][row["metric"]] = float(val) if val != "" else None
    return data


def retained_tasks(data, model, condition):
    """Task ids retained under listwise deletion for a (model, condition) family.

    A task is kept only if all six architectures yield a usable value; for the
    `passing` condition every architecture must additionally pass the tests.
    (Validity is metric-independent here -- radon produces all nine metrics or
    none -- so the retained set is shared across metrics within a condition.)
    """
    tasks = sorted(data[(model, ARCHS[0])].keys(),
                   key=lambda t: int(t.rsplit("/", 1)[-1]))
    kept = []
    for t in tasks:
        ok = True
        for a in ARCHS:
            cell = data[(model, a)][t]
            if not cell["valid"] or any(cell["m"][m] is None for m in TEST_METRICS):
                ok = False
                break
            if condition == "passing" and not cell["passed"]:
                ok = False
                break
        if ok:
            kept.append(t)
    return kept


def matrix(data, model, metric, tasks):
    """n x 6 array of `metric` values, columns ordered as ARCHS."""
    return np.array([[data[(model, a)][t]["m"][metric] for a in ARCHS]
                     for t in tasks], dtype=float)


# --------------------------------------------------------------------------- #
# Effect-size helpers
# --------------------------------------------------------------------------- #
def kendall_w(q, n):
    return q / (n * DF) if n > 0 else float("nan")


def w_magnitude(w):
    if np.isnan(w):
        return "n/a"
    return "weak" if w < 0.3 else ("moderate" if w < 0.5 else "strong")


def rrb_magnitude(r):
    if r is None or np.isnan(r):
        return "n/a"
    a = abs(r)
    return "negligible" if a < 0.1 else ("small" if a < 0.3 else
           ("moderate" if a < 0.5 else "large"))


def rank_biserial(a_vals, b_vals):
    """Matched-pairs rank-biserial r for paired samples a vs b.

    Positive => a tends to exceed b. Returns None if every pair is tied.
    """
    d = np.asarray(a_vals, float) - np.asarray(b_vals, float)
    d = d[d != 0]                       # Wilcoxon 'wilcox' zero handling
    if d.size == 0:
        return None
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    return (w_plus - w_minus) / total if total > 0 else None


def mean_ranks(mat):
    """Friedman mean rank per architecture (column). Lower rank => smaller value."""
    per_row = np.array([stats.rankdata(row) for row in mat])
    return per_row.mean(axis=0)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main():
    warnings.simplefilter("ignore")  # benign scipy divide warnings on all-tie pairs
    data = load_data()

    omnibus_rows, posthoc_rows, meanrank_rows, desc_rows = [], [], [], []

    # ---- retained-task sets per (model, condition) -------------------------
    retained = {}
    for model in MODELS:
        for cond in ("primary", "passing"):
            retained[(model, cond)] = retained_tasks(data, model, cond)

    # ---- descriptive statistics (all 9 metrics) ---------------------------
    for model in MODELS:
        for cond in ("primary", "passing"):
            tasks = retained[(model, cond)]
            for arch in ARCHS:
                for metric in DESC_METRICS:
                    vals = np.array([data[(model, arch)][t]["m"][metric]
                                     for t in tasks], dtype=float)
                    desc_rows.append({
                        "model": model, "architecture": arch, "condition": cond,
                        "metric": metric, "n": len(vals),
                        "mean": vals.mean() if vals.size else float("nan"),
                        "sd": vals.std(ddof=1) if vals.size > 1 else float("nan"),
                        "median": np.median(vals) if vals.size else float("nan"),
                        "q1": np.percentile(vals, 25) if vals.size else float("nan"),
                        "q3": np.percentile(vals, 75) if vals.size else float("nan"),
                        "min": vals.min() if vals.size else float("nan"),
                        "max": vals.max() if vals.size else float("nan"),
                    })

    # ---- omnibus + post-hoc ----------------------------------------------
    for model in MODELS:
        for cond in ("primary", "passing"):
            tasks = retained[(model, cond)]
            n = len(tasks)
            for metric in TEST_METRICS:
                mat = matrix(data, model, metric, tasks)

                # mean ranks (reported regardless of inferential outcome)
                if n > 0:
                    mr = mean_ranks(mat)
                    for j, arch in enumerate(ARCHS):
                        meanrank_rows.append({
                            "model": model, "metric": metric, "condition": cond,
                            "architecture": arch, "n": n, "mean_rank": mr[j],
                        })

                testable = n >= MIN_N
                q = p = w = float("nan")
                significant = False
                if testable:
                    q, p = stats.friedmanchisquare(*[mat[:, j] for j in range(K)])
                    w = kendall_w(q, n)
                    significant = p < ALPHA

                omnibus_rows.append({
                    "model": model, "metric": metric, "condition": cond,
                    "n": n, "friedman_q": q, "df": DF, "p_value": p,
                    "significant": significant, "kendall_w": w,
                    "w_magnitude": w_magnitude(w),
                    "tested": testable,
                    "note": "" if testable else f"n<{MIN_N}: inferential test omitted",
                })

                # post-hoc only where the omnibus rejects
                if not (testable and significant):
                    continue

                p_raw, pair_meta = [], []
                for a, b, layers, mech in PAIRS:
                    av, bv = mat[:, ARCH_IDX[a]], mat[:, ARCH_IDX[b]]
                    if np.all(av - bv == 0):
                        pr = 1.0
                        wstat = float("nan")
                    else:
                        res = stats.wilcoxon(av, bv)  # two-sided, drop zeros
                        wstat, pr = float(res.statistic), float(res.pvalue)
                    p_raw.append(pr)
                    pair_meta.append((a, b, layers, mech, wstat,
                                      rank_biserial(av, bv),
                                      float(np.median(av) - np.median(bv))))

                reject, p_holm, _, _ = multipletests(p_raw, alpha=ALPHA, method="holm")
                for (a, b, layers, mech, wstat, rrb, mdiff), pr, ph, rej in zip(
                        pair_meta, p_raw, p_holm, reject):
                    if rrb is not None and not np.isnan(rrb):
                        higher = a if rrb > 0 else (b if rrb < 0 else "tie")
                    else:
                        higher = a if mdiff > 0 else (b if mdiff < 0 else "tie")
                    posthoc_rows.append({
                        "model": model, "metric": metric, "condition": cond,
                        "arch_a": a, "arch_b": b, "layers": layers,
                        "mechanism": mech, "n": n, "wilcoxon_stat": wstat,
                        "p_raw": pr, "p_holm": ph, "significant": bool(rej),
                        "rank_biserial": rrb if rrb is not None else float("nan"),
                        "rrb_magnitude": rrb_magnitude(rrb),
                        "median_diff": mdiff, "higher_complexity": higher,
                    })

    # ---- write CSVs -------------------------------------------------------
    def write(name, rows, fields):
        with open(out_path(name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)

    write("stats_omnibus.csv", omnibus_rows,
          ["model", "metric", "condition", "n", "friedman_q", "df", "p_value",
           "significant", "kendall_w", "w_magnitude", "tested", "note"])
    write("stats_posthoc.csv", posthoc_rows,
          ["model", "metric", "condition", "arch_a", "arch_b", "layers",
           "mechanism", "n", "wilcoxon_stat", "p_raw", "p_holm", "significant",
           "rank_biserial", "rrb_magnitude", "median_diff", "higher_complexity"])
    write("stats_meanranks.csv", meanrank_rows,
          ["model", "metric", "condition", "architecture", "n", "mean_rank"])
    write("stats_descriptive.csv", desc_rows,
          ["model", "architecture", "condition", "metric", "n", "mean", "sd",
           "median", "q1", "q3", "min", "max"])

    print_summary(retained, omnibus_rows, posthoc_rows)

    run_directional_tests(data, retained)


def run_directional_tests(data, retained):
    """Pre-registered directional test: does adding the Debugger layer (D)
    produce *more* complex code, on backgrounds that isolate D? One-sided
    Wilcoxon signed-rank (alternative='greater') comparing the D-config to
    the matched non-D-config, per (model, metric). Primary condition only.
    """
    d_pairs = [
        ("Basic", "Debugger"),
        ("AC", "AC+Debugger"),
        ("ACT", "ACT+Debugger"),
    ]
    rows = []
    for model in MODELS:
        tasks = retained[(model, "primary")]
        n = len(tasks)
        for metric in TEST_METRICS:
            for non_d, d_config in d_pairs:
                nond_vals = np.array(
                    [data[(model, non_d)][t]["m"][metric] for t in tasks],
                    dtype=float,
                )
                d_vals = np.array(
                    [data[(model, d_config)][t]["m"][metric] for t in tasks],
                    dtype=float,
                )
                if n == 0 or np.all(d_vals - nond_vals == 0):
                    p_one = 1.0
                else:
                    res = stats.wilcoxon(d_vals, nond_vals, alternative="greater")
                    p_one = float(res.pvalue)
                mdiff = (float(np.median(d_vals) - np.median(nond_vals))
                         if n > 0 else float("nan"))
                rows.append({
                    "model": model,
                    "metric": metric,
                    "condition": "primary",
                    "non_d": non_d,
                    "d_config": d_config,
                    "n": n,
                    "p_one_sided": p_one,
                    "median_diff_d_minus_nond": mdiff,
                    "direction_supported": p_one < 0.05,
                })
    fields = ["model", "metric", "condition", "non_d", "d_config", "n",
              "p_one_sided", "median_diff_d_minus_nond", "direction_supported"]
    out = out_path("stats_directional.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"written: {out}  ({len(rows)} rows)")


def print_summary(retained, omnibus_rows, posthoc_rows):
    bar = "=" * 78
    print("\n" + bar + "\nSTATISTICAL PIPELINE SUMMARY\n" + bar)
    for model in MODELS:
        print(f"retained n -- {model}:  primary = {len(retained[(model,'primary')])}"
              f"   passing-only = {len(retained[(model,'passing')])}")

    for cond in ("primary", "passing"):
        print(f"\n{'-'*78}\nOMNIBUS  (Friedman)  --  condition: {cond}\n{'-'*78}")
        print(f"{'model':<13}{'metric':<22}{'n':>5}{'Q':>10}{'p':>12}"
              f"{'W':>8}  {'sig':<5}{'effect'}")
        for r in omnibus_rows:
            if r["condition"] != cond:
                continue
            sig = "YES" if r["significant"] else "no"
            q = f"{r['friedman_q']:.2f}" if r["tested"] else "--"
            p = f"{r['p_value']:.2e}" if r["tested"] else "--"
            w = f"{r['kendall_w']:.3f}" if r["tested"] else "--"
            print(f"{r['model']:<13}{r['metric']:<22}{r['n']:>5}{q:>10}{p:>12}"
                  f"{w:>8}  {sig:<5}{r['w_magnitude']}")

    print(f"\n{'-'*78}\nPOST-HOC  --  significant pairs only "
          f"(Holm-adjusted p < {ALPHA})\n{'-'*78}")
    for cond in ("primary", "passing"):
        for model in MODELS:
            for metric in TEST_METRICS:
                sig = [r for r in posthoc_rows
                       if r["condition"] == cond and r["model"] == model
                       and r["metric"] == metric and r["significant"]]
                if not sig:
                    continue
                print(f"[{cond}] {model} / {metric}: "
                      f"{len(sig)}/15 pairs significant")
                for r in sig:
                    print(f"    {r['arch_a']:>13} vs {r['arch_b']:<13} "
                          f"({r['mechanism']:<8} {r['layers']:<7}) "
                          f"p_holm={r['p_holm']:.1e}  r_rb={r['rank_biserial']:+.2f}"
                          f" ({r['rrb_magnitude']})  higher: {r['higher_complexity']}")

    print(bar)
    print("written: stats_omnibus.csv, stats_posthoc.csv, stats_meanranks.csv, "
          "stats_descriptive.csv")
    print(bar)


if __name__ == "__main__":
    main()
