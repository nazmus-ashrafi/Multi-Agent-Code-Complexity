# Multi-Agent Code Complexity — Replication Package

Replication materials for:

> **How Generation Architecture Shapes Code Complexity in Multi-Agent LLM Systems: A Paired Study on HumanEval**
>
> Nazmus Ashrafi · [arXiv:2606.00308](https://arxiv.org/abs/2606.00308)

The study measures how the multi-agent **generation architecture** an LLM is wrapped
in affects the **structural complexity** of the code it produces. Six widely-used
configurations are compared across two models (GPT-4o and GPT-4o-mini) on all 164
HumanEval tasks, using the five `radon` complexity metrics and a paired
non-parametric statistical pipeline (Friedman → Wilcoxon signed-rank with Holm
correction; Kendall's *W* and matched-pairs rank-biserial effect sizes).

This package contains everything needed to reproduce the paper's data, statistics,
tables, and figures from scratch, and ships the exact generated solutions used in
the paper so the analysis can be reproduced **without** spending API credits.

**Just want to see the results?** No setup needed — the three result tables are
committed as Markdown and render directly on GitHub at
[`analysis/tables/`](analysis/tables/), and the five figures are in
[`analysis/figures/`](analysis/figures/).

## The six architectures

| Flow (`--flow`) | Paper name     | Layers            |
|-----------------|----------------|-------------------|
| `basic`         | `Basic`        | —                 |
| `AC`            | `AC`           | R                 |
| `ACT`           | `ACT`          | R + T             |
| `debugger`      | `Debugger`     | D                 |
| `ac_debugger`   | `AC+Debugger`  | R + D             |
| `act_debugger`  | `ACT+Debugger` | R + T + D         |

R = role decomposition (Analyst), T = testing with bounded iteration (Tester),
D = runtime debugging (CFG-based Debugger).

## Repository structure

```
.
├── main.py                  # Generation entry point: one (model, flow) run on HumanEval
├── utils.py
├── agents/                  # Analyst, Coder, Tester agents
├── flows/                   # The six configuration flows (+ debugger / execution utils)
├── staticfg/                # Control-flow-graph builder used by the debugger flows
├── evaluation/              # HumanEval functional-correctness harness (adapted from OpenAI human-eval)
├── data/                    # HumanEval data files
├── outputs/                 # Generated solutions used in the paper (committed)
│   ├── gpt4o_generations/        # *.jsonl (generations) + *_results.jsonl (pass@1)
│   └── gpt4omini_generations/
├── analysis/                # Metric extraction, statistics, tables, figures
│   ├── extract_metrics.py        # outputs/ -> analysis/complexity_metrics.csv
│   ├── run_stats.py              # complexity_metrics.csv -> stats_*.csv
│   ├── make_tables.py            # stats_*.csv -> Markdown tables (analysis/tables/)
│   ├── make_figures.py           # -> figure PNGs (analysis/figures/)
│   ├── complexity_metrics.csv    # committed: per-completion radon metrics
│   ├── stats_*.csv               # committed: omnibus / posthoc / meanranks / descriptive / directional
│   ├── tables/                   # committed: the 3 result tables as Markdown (viewable on GitHub)
│   └── figures/                  # committed: PNG previews of the five paper figures
├── pyproject.toml / uv.lock # Pinned dependencies
└── .env.example             # Copy to .env and add your API keys
```

## Setup

Requires Python ≥ 3.13. Pick **one** of the two setups below, then **activate the
environment** so that `python` resolves to the project interpreter (not your system
or conda `base` Python).

**With uv** (recommended; uses the pinned `uv.lock`):

```bash
uv sync
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Without uv** (standard venv + pip):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

> If you see `ModuleNotFoundError: No module named 'radon'`, the environment is not
> active and `python` is pointing at a different interpreter. Either activate the
> venv as above, or prefix each command with `uv run` (e.g. `uv run python
> analysis/extract_metrics.py`).

Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` (only needed to
regenerate solutions; the committed generations let you skip this and go straight
to the analysis).

## Reproducing the results

### Option A — reproduce analysis from the committed generations (no API needed)

```bash
python analysis/extract_metrics.py     # outputs/ -> analysis/complexity_metrics.csv
python analysis/run_stats.py           # -> analysis/stats_omnibus.csv, stats_posthoc.csv, ...
python analysis/make_tables.py         # -> analysis/tables/*.md
python analysis/make_figures.py        # -> analysis/figures/*.png
```

`make_tables.py` writes the three result tables as **Markdown** (in
`analysis/tables/`, readable on GitHub), and `make_figures.py` writes the five
figures as PNGs (in `analysis/figures/`).

### Option B — regenerate solutions from scratch (uses API credits)

This path has two steps per (model, flow): **generate** the code, then **evaluate**
it for functional correctness. Generation does *not* compute pass@1 — it only writes
the completions; the pass@1 `*_results.jsonl` files come from a separate run of the
HumanEval harness, and the analysis (Option A) needs them.

**Step 1 — generate** (repeat for all six flows × both models):

```bash
python main.py \
  --provider_and_model openai:gpt-4o-2024-08-06 \
  --flow basic \
  --range full \
  --output_path outputs/gpt4o_generations/gpt4o_basic_humaneval.jsonl
```

Models used in the paper: `openai:gpt-4o-2024-08-06` and
`openai:gpt-4o-mini-2024-07-18`, decoding at temperature 0. HumanEval is loaded via
the Hugging Face `openai_humaneval` dataset. (OpenAI does not guarantee bit-identical
outputs at temperature 0, so a fresh run may differ slightly from the committed
generations.)

**Step 2 — evaluate** each generated file for pass@1. The harness is run from inside
`evaluation/` and writes `<sample_file>_results.jsonl` next to the input:

```bash
cd evaluation
python evaluate_functional_correctness.py \
  --sample_file ../outputs/gpt4o_generations/gpt4o_basic_humaneval.jsonl \
  --problem_file data/HumanEval.jsonl.gz
cd ..
# -> outputs/gpt4o_generations/gpt4o_basic_humaneval.jsonl_results.jsonl
```

The sample file must contain a completion for all 164 HumanEval tasks (the harness
asserts completeness).

> ⚠️ **Security:** the harness executes untrusted, model-generated code via `exec`.
> Run it only inside a sandbox / disposable environment.

Once every generation file has its matching `*_results.jsonl`, run the Option A steps
to rebuild the metrics, statistics, tables, and figures.

**On reproducibility of pass@1.** The committed `*_results.jsonl` are the authoritative
pass@1 used in the paper, and the Option A analysis reproduces the paper's numbers
exactly from them. Re-running the evaluation harness in a different environment can
shift pass@1 by a task or two (different Python/library versions, execution timeouts),
consistent with the non-determinism the paper notes in its threats to validity. The
structural-complexity results are computed directly from the generations and are
unaffected by such small pass@1 differences.

## Notes

- The committed generations are the final runs used in the paper (post the
  docstring-preserving extraction fix discussed in the paper's threats to validity);
  the superseded earlier run is not included here.
- `evaluation/` adapts OpenAI's HumanEval harness (MIT); `staticfg/` is the staticfg
  CFG library. Both retain their own LICENSE files.

## Timestamping

Every push to `main` is timestamped with [OpenTimestamps](https://opentimestamps.org/)
via a GitHub Action (`.github/workflows/timestamp.yml`). It stamps the HEAD commit
hash — which covers the full repository tree and history — and commits the proof into
`ots-proofs/`. Each proof is a trustless, Bitcoin-anchored record of when that state of
the repository existed. Verify a proof with:

```bash
ots verify ots-proofs/<commit-hash>.txt.ots
```

(Proofs are "pending" for a few hours until a Bitcoin block attests them; run
`ots upgrade ots-proofs/<commit-hash>.txt.ots` to fetch the completed proof.)

## License

MIT — see [LICENSE](LICENSE).
