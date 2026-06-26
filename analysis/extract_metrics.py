#!/usr/bin/env python3
"""
extract_metrics.py -- RADON complexity-metric extraction for the multi-agent code complexity study.

Reads the 12 HumanEval generation files (2 models x 6 architectures), computes
RADON complexity metrics on the *model-generated completion only*, records two
data-quality flags, joins the reference-test pass outcome, and emits a tidy
(long-format) CSV that feeds the paired Friedman/Wilcoxon statistical pipeline.

Measurement scope follows the Methodology section: metrics are computed on the
`completion` field (already fence-stripped upstream by clean_code_function in
utils.py) and exclude the HumanEval prompt scaffolding and the hidden tests.

Output CSV columns (long format, one row per task x model x architecture x metric):
    task_id, model, architecture, metric, value, passed,
    parse_valid, entry_point_present

Metrics (9):
    loc, sloc, comments, multi, blank   -- radon.raw.analyze
    cc                                   -- radon.complexity.cc_visit
    halstead_volume/difficulty/effort    -- radon.metrics.h_visit

Aggregation choices (the Methodology defines the metrics but not their
per-completion aggregation; both choices below are documented here so the paper
can state them precisely):
    - Halstead V/D/E: radon's module-level `total`, i.e. computed over the whole
      completion at once. No aggregation ambiguity.
    - CC: SUM of per-block cyclomatic complexity over every function/class in the
      completion -- the program-level McCabe total. For the ~96% of completions
      that contain a single function this equals that function's CC; it differs
      only for multi-function completions (~4% of the corpus).

Uncomputable cells (parse-invalid completion, or radon raising on parse-valid
code) are still emitted as rows, with an empty `value` and the flags recording
the cause, so downstream listwise deletion sees the complete grid.
"""
import ast
import csv
import json
import os
import sys

import radon.raw as radon_raw
import radon.complexity as radon_cx
import radon.metrics as radon_met

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(REPO, "outputs")
OUT_CSV = os.path.join(REPO, "analysis", "complexity_metrics.csv")

# Explicit file -> (model, architecture) map. The directory layout is asymmetric
# (gpt-4o files in outputs/gpt4o_generations/, gpt-4o-mini in
# outputs/gpt4omini_generations/) and filename conventions are inconsistent --
# some files carry a `_humaneval` suffix, some do not -- so an explicit map is
# required rather than pattern-matching.
FILE_MAP = [
    # (path relative to outputs/, model, architecture)
    ("gpt4o_generations/gpt4o_basic_humaneval.jsonl",          "gpt-4o",      "Basic"),
    ("gpt4o_generations/gpt4o_ac_humaneval.jsonl",             "gpt-4o",      "AC"),
    ("gpt4o_generations/gpt4o_act_humaneval.jsonl",            "gpt-4o",      "ACT"),
    ("gpt4o_generations/gpt4o_debugger_humaneval.jsonl",       "gpt-4o",      "Debugger"),
    ("gpt4o_generations/gpt4o_acdebugger_humaneval.jsonl",     "gpt-4o",      "AC+Debugger"),
    ("gpt4o_generations/gpt4o_actdebugger_humaneval.jsonl",    "gpt-4o",      "ACT+Debugger"),
    ("gpt4omini_generations/gpt4omini_basic_humaneval.jsonl",      "gpt-4o-mini", "Basic"),
    ("gpt4omini_generations/gpt4omini_ac.jsonl",                   "gpt-4o-mini", "AC"),
    ("gpt4omini_generations/gpt4omini_act.jsonl",                  "gpt-4o-mini", "ACT"),
    ("gpt4omini_generations/gpt4omini_debugger.jsonl",             "gpt-4o-mini", "Debugger"),
    ("gpt4omini_generations/gpt4omini_ac_debugger.jsonl",          "gpt-4o-mini", "AC+Debugger"),
    ("gpt4omini_generations/gpt4omini_actdebugger_humaneval.jsonl","gpt-4o-mini", "ACT+Debugger"),
]

METRIC_NAMES = [
    "loc", "sloc", "comments", "multi", "blank", "cc",
    "halstead_volume", "halstead_difficulty", "halstead_effort",
]

# Expected per-(model, architecture) test-pass counts out of 164, derived from
# the locked pass@1 rates in the handoff briefing. Used purely as a join /
# file-map integrity check at run time.
EXPECTED_PASS = {
    ("gpt-4o-mini", "Basic"): 139, ("gpt-4o-mini", "AC"): 143,
    ("gpt-4o-mini", "ACT"): 138,   ("gpt-4o-mini", "Debugger"): 144,
    ("gpt-4o-mini", "AC+Debugger"): 144, ("gpt-4o-mini", "ACT+Debugger"): 142,
    ("gpt-4o", "Basic"): 144, ("gpt-4o", "AC"): 145,
    ("gpt-4o", "ACT"): 147,   ("gpt-4o", "Debugger"): 151,
    ("gpt-4o", "AC+Debugger"): 151, ("gpt-4o", "ACT+Debugger"): 145,
}


def read_jsonl(path):
    """Read a .jsonl file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_num(task_id):
    """Numeric sort key for 'HumanEval/<n>' task ids."""
    return int(str(task_id).rsplit("/", 1)[-1])


def entry_point_present(tree, entry_point):
    """True iff the module body defines a top-level function named entry_point.

    The test harness calls the entry point as a module-level name, so a function
    nested inside a class or another function would not be a usable entry point;
    the check is therefore restricted to the top-level module body.
    """
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point
        for n in tree.body
    )


def compute_metrics(code):
    """Return a dict of the 9 RADON metrics for `code`.

    Assumes `code` is parse-valid (caller checks ast.parse first). Raises if
    radon itself fails; the caller catches that separately from a parse failure.
    """
    raw = radon_raw.analyze(code)
    blocks = radon_cx.cc_visit(code)
    halstead = radon_met.h_visit(code).total
    return {
        "loc": raw.loc,
        "sloc": raw.sloc,
        "comments": raw.comments,
        "multi": raw.multi,
        "blank": raw.blank,
        # Program-level McCabe total: sum over every function/class block.
        "cc": sum(b.complexity for b in blocks),
        "halstead_volume": halstead.volume,
        "halstead_difficulty": halstead.difficulty,
        "halstead_effort": halstead.effort,
    }


def process_file(rel_path, model, architecture):
    """Process one (model, architecture) generation file.

    Returns (rows, stats) where rows is the list of long-format CSV records and
    stats is a per-file summary dict.
    """
    gen_path = os.path.join(OUTPUTS, rel_path)
    res_path = gen_path + "_results.jsonl"
    if not os.path.exists(gen_path):
        sys.exit(f"ERROR: missing generation file: {gen_path}")
    if not os.path.exists(res_path):
        sys.exit(f"ERROR: missing results file: {res_path}")

    gen_rows = read_jsonl(gen_path)
    res_rows = read_jsonl(res_path)

    # Join the test outcome by task_id. The results file is the eval harness's
    # copy of the generation file plus `result`/`passed`; we still read both and
    # cross-check the completion text so a silent mismatch surfaces.
    res_by_id = {r["task_id"]: r for r in res_rows}
    if set(res_by_id) != {g["task_id"] for g in gen_rows}:
        sys.exit(f"ERROR: task_id mismatch between {gen_path} and {res_path}")

    rows = []
    n_parse_invalid = n_ep_missing = n_radon_fail = n_passed = n_multiblock = 0
    n_completion_mismatch = 0

    for g in gen_rows:
        task_id = g["task_id"]
        entry = g.get("entry_point")
        completion = g.get("completion") or ""
        res = res_by_id[task_id]
        passed = bool(res.get("passed"))
        if passed:
            n_passed += 1
        if (res.get("completion") or "") != completion:
            n_completion_mismatch += 1

        # --- data-quality flags ---
        try:
            tree = ast.parse(completion)
            parse_valid = True
        except SyntaxError:
            tree = None
            parse_valid = False
            n_parse_invalid += 1

        ep_ok = bool(tree is not None and entry_point_present(tree, entry))
        if not ep_ok:
            n_ep_missing += 1

        # --- metrics ---
        metrics = None
        if parse_valid:
            try:
                metrics = compute_metrics(completion)
                if len(radon_cx.cc_visit(completion)) > 1:
                    n_multiblock += 1
            except Exception as exc:  # radon failed on parse-valid code
                n_radon_fail += 1
                print(f"  WARN radon failed on {model}/{architecture} {task_id}: "
                      f"{type(exc).__name__}: {exc}")

        for metric in METRIC_NAMES:
            value = "" if metrics is None else metrics[metric]
            rows.append({
                "task_id": task_id,
                "model": model,
                "architecture": architecture,
                "metric": metric,
                "value": value,
                "passed": passed,
                "parse_valid": parse_valid,
                "entry_point_present": ep_ok,
            })

    stats = {
        "model": model, "architecture": architecture,
        "n_tasks": len(gen_rows),
        "parse_invalid": n_parse_invalid,
        "ep_missing": n_ep_missing,
        "radon_fail": n_radon_fail,
        "multiblock": n_multiblock,
        "passed": n_passed,
        "completion_mismatch": n_completion_mismatch,
    }
    return rows, stats


def main():
    all_rows = []
    all_stats = []
    for rel_path, model, architecture in FILE_MAP:
        rows, stats = process_file(rel_path, model, architecture)
        all_rows.extend(rows)
        all_stats.append(stats)

    # Stable ordering: model, architecture (file-map order), task, metric.
    arch_order = {a: i for i, (_, _, a) in enumerate(
        [fm for fm in FILE_MAP if fm[1] == "gpt-4o"])}
    model_order = {"gpt-4o": 0, "gpt-4o-mini": 1}
    metric_order = {m: i for i, m in enumerate(METRIC_NAMES)}
    all_rows.sort(key=lambda r: (
        model_order[r["model"]], arch_order[r["architecture"]],
        task_num(r["task_id"]), metric_order[r["metric"]],
    ))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["task_id", "model", "architecture", "metric", "value",
              "passed", "parse_valid", "entry_point_present"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- run summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXTRACTION SUMMARY")
    print("=" * 72)
    hdr = f"{'model':<13}{'architecture':<15}{'tasks':>6}{'parse!':>8}" \
          f"{'ep-miss':>9}{'radon!':>8}{'multiF':>8}{'passed':>8}{'pass%':>8}"
    print(hdr)
    print("-" * len(hdr))
    tot_parse = tot_ep = tot_radon = tot_mismatch = 0
    pass_ok = True
    for s in all_stats:
        rate = 100.0 * s["passed"] / s["n_tasks"]
        exp = EXPECTED_PASS.get((s["model"], s["architecture"]))
        flag = ""
        if exp is not None and exp != s["passed"]:
            flag = f"  <-- EXPECTED {exp}"
            pass_ok = False
        print(f"{s['model']:<13}{s['architecture']:<15}{s['n_tasks']:>6}"
              f"{s['parse_invalid']:>8}{s['ep_missing']:>9}{s['radon_fail']:>8}"
              f"{s['multiblock']:>8}{s['passed']:>8}{rate:>7.2f}%{flag}")
        tot_parse += s["parse_invalid"]
        tot_ep += s["ep_missing"]
        tot_radon += s["radon_fail"]
        tot_mismatch += s["completion_mismatch"]
    print("-" * len(hdr))
    print(f"totals: parse-invalid={tot_parse}  entry-point-missing={tot_ep}  "
          f"radon-failures={tot_radon}  gen/results completion mismatches={tot_mismatch}")
    print(f"rows written: {len(all_rows)}  (expected {len(FILE_MAP)*164*len(METRIC_NAMES)})")
    print(f"output: {OUT_CSV}")
    if pass_ok:
        print("pass-count check: OK -- all 12 cells match expected pass@1 counts")
    else:
        print("pass-count check: MISMATCH -- file map or join is wrong (see flags above)")
    print("=" * 72)


if __name__ == "__main__":
    main()
