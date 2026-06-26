#!/usr/bin/env python3
"""
make_tables.py -- result tables for the multi-agent code complexity study.

Reads the CSVs produced by run_stats.py and emits the three result tables as
Markdown into analysis/tables/, readable directly on GitHub (no LaTeX needed):

  tab_descriptive.md   median [Q1, Q3] per (model, architecture, metric),
                       all-completions condition
  tab_omnibus.md       Friedman Q and Kendall's W for all 20 omnibus tests
  tab_posthoc.md       pairwise rank-biserial for the 15 comparisons (SLOC;
                       the significance pattern is metric-invariant)
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "analysis", "tables")

MODELS = ["gpt-4o", "gpt-4o-mini"]
ARCHS = ["Basic", "AC", "ACT", "Debugger", "AC+Debugger", "ACT+Debugger"]
TEST_METRICS = ["sloc", "cc", "halstead_volume", "halstead_difficulty", "halstead_effort"]
METRIC_HDR = {"sloc": "SLOC", "cc": "CC", "halstead_volume": "Halstead V",
              "halstead_difficulty": "Halstead D", "halstead_effort": "Halstead E"}
# decimal places for the median/IQR display per metric
DECIMALS = {"sloc": 0, "cc": 0, "halstead_volume": 0,
            "halstead_difficulty": 1, "halstead_effort": 0}


def load(name):
    with open(os.path.join(REPO, "analysis", name)) as f:
        return list(csv.DictReader(f))


def fmt(x, dec):
    return f"{x:.{dec}f}" if dec else f"{round(x)}"


def md_table(headers, rows):
    """Render a Markdown table from a header list and a list of row lists."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def table_descriptive(desc):
    rows = {(r["model"], r["architecture"], r["metric"]): r
            for r in desc if r["condition"] == "primary"}
    L = ["# Descriptive complexity statistics", "",
         "Median [Q1, Q3] per (model, architecture) cell over the all-completions "
         "condition (n = 164 tasks per cell). Lean cluster: Basic, Debugger, "
         "AC+Debugger; heavy cluster: AC, ACT, ACT+Debugger.", ""]
    for model in MODELS:
        L += [f"### {model}", ""]
        headers = ["Architecture"] + [METRIC_HDR[m] for m in TEST_METRICS]
        body = []
        for arch in ARCHS:
            cells = [arch]
            for m in TEST_METRICS:
                r = rows[(model, arch, m)]
                d = DECIMALS[m]
                cells.append(f"{fmt(float(r['median']), d)} "
                             f"[{fmt(float(r['q1']), d)}, {fmt(float(r['q3']), d)}]")
            body.append(cells)
        L += [md_table(headers, body), ""]
    return "\n".join(L) + "\n"


def table_omnibus(omni):
    o = {(r["model"], r["condition"], r["metric"]): r for r in omni}
    L = ["# Omnibus Friedman tests", "",
         "Statistic Q (df = 5) and Kendall's concordance W. All twenty tests "
         "reject the null of equal complexity across the six architectures "
         "(every p < 1e-20). Primary = all-completions; Passing = passing-only.", ""]
    for model in MODELS:
        L += [f"### {model}", ""]
        headers = ["Metric", "Primary Q", "Primary W", "Passing Q", "Passing W"]
        body = []
        for metric in TEST_METRICS:
            p, pa = o[(model, "primary", metric)], o[(model, "passing", metric)]
            body.append([METRIC_HDR[metric],
                         f"{float(p['friedman_q']):.1f}", f"{float(p['kendall_w']):.3f}",
                         f"{float(pa['friedman_q']):.1f}", f"{float(pa['kendall_w']):.3f}"])
        L += [md_table(headers, body), ""]
    return "\n".join(L) + "\n"


def table_posthoc(posthoc, metric="sloc"):
    rows = [r for r in posthoc if r["metric"] == metric]
    by = {(r["model"], r["condition"], r["arch_a"], r["arch_b"]): r for r in rows}
    pairs, seen = [], set()
    for r in rows:
        key = (r["arch_a"], r["arch_b"])
        if key not in seen:
            seen.add(key)
            pairs.append((r["arch_a"], r["arch_b"], r["layers"], r["mechanism"]))

    def cell(model, cond, a, b):
        r = by[(model, cond, a, b)]
        s = f"{float(r['rank_biserial']):+.2f}"
        return f"**{s}**" if r["significant"] == "True" else s

    L = ["# Post-hoc pairwise comparisons", "",
         "Matched-pairs rank-biserial correlation r_rb (SLOC; the Holm-significance "
         "pattern is identical for all five metrics). Positive r_rb means the first "
         "architecture is the more complex. **Bold** entries are significant after "
         "Holm correction within the 15-pair family (p < 0.05).", ""]
    headers = ["Comparison", "Layer(s)", "Type",
               "gpt-4o Primary", "gpt-4o Passing",
               "gpt-4o-mini Primary", "gpt-4o-mini Passing"]
    body = []
    for a, b, layers, mech in pairs:
        body.append([f"{a} vs {b}", layers.replace("<->", "↔"), mech,
                     cell("gpt-4o", "primary", a, b), cell("gpt-4o", "passing", a, b),
                     cell("gpt-4o-mini", "primary", a, b), cell("gpt-4o-mini", "passing", a, b)])
    L += [md_table(headers, body), ""]
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    desc = load("stats_descriptive.csv")
    omni = load("stats_omnibus.csv")
    posthoc = load("stats_posthoc.csv")

    files = {
        "tab_descriptive.md": table_descriptive(desc),
        "tab_omnibus.md": table_omnibus(omni),
        "tab_posthoc.md": table_posthoc(posthoc),
    }
    for name, body in files.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"  wrote {os.path.relpath(path, REPO)}")
    print(f"\n{len(files)} Markdown tables written to analysis/tables/.")


if __name__ == "__main__":
    main()
