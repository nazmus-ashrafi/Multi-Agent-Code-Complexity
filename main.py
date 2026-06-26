"""
Multi-Agent Complexities Project — Main Entry Point
=====================================================
Runs 6 multi-agent code generation configurations on HumanEval
for the complexity analysis study (extension of AICT 2025 paper).

Models used in the paper:
  - openai:gpt-4o-2024-08-06        (previous-generation flagship)
  - openai:gpt-4o-mini-2024-07-18   (cost-efficient sibling)

Supported flows:
  1. basic          — Single-shot prompting
  2. AC             — Analyst + Coder
  3. ACT            — Analyst + Coder + Tester (up to 3 iterations)
  4. debugger       — Runtime debugging only (CFG-based)
  5. ac_debugger    — AC + runtime debugging
  6. act_debugger   — ACT + runtime debugging (full LDB)

Usage:
  python main.py --provider_and_model openai:gpt-4o-2024-08-06 --flow basic --range full
  python main.py --provider_and_model openai:gpt-4o-mini-2024-07-18 --flow ACT --range 0:10
"""

import os
import json
import argparse
import tqdm

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, DatasetDict

from utils import prompt_split_humaneval
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from agents.coder_agent import CoderAgent
from agents.analyst_agent import AnalystAgent
from agents.tester_agent import TesterAgent

from flows.flow import Flow
from flows.analyst_coder_flow import Analyst_Coder_Flow
from flows.analyst_coder_tester_flow import Analyst_Coder_Tester_Flow
from flows.llm_debugger_flow import LDB_Flow
from flows.debugger_only_flow import Debugger_Only_Flow
from flows.AC_Debug_flow import AC_Debug_Flow


# ─────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Multi-Agent Complexities Project — HumanEval Experiment Runner"
)
parser.add_argument('--output_path', type=str, default='outputs/output.jsonl',
                    help="Path to the output JSONL file")
parser.add_argument('--range', type=str, default='full',
                    help="Dataset range: 'full' or 'start:end' (e.g., '0:10')")
parser.add_argument('--provider_and_model', type=str, required=True,
                    help="Provider:model (e.g., 'openai:gpt-4o-mini-2024-07-18')")
parser.add_argument('--flow', type=str, required=True,
                    choices=['basic', 'AC', 'ACT', 'debugger', 'ac_debugger', 'act_debugger'],
                    help="Configuration to run")
parser.add_argument('--api_key', type=str, default=None,
                    help="API key (falls back to env vars if not provided)")

args = parser.parse_args()


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def create_model(provider: str, model_name: str, api_key: str = None):
    """Create a LangChain chat model from provider and model name."""

    if provider == "openai":
        kwargs = {"model_name": model_name, "temperature": 0}
        if api_key:
            kwargs["openai_api_key"] = api_key
        return ChatOpenAI(**kwargs)

    elif provider == "groq":
        kwargs = {"model_name": model_name, "temperature": 0}
        if api_key:
            kwargs["groq_api_key"] = api_key
        return ChatGroq(**kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"This project supports: openai, groq"
        )


# ─────────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────────

def create_agents(model):
    """Create all agent instances needed across the six flows."""
    return {
        "basic_coder":    CoderAgent(model=model, flow="basic"),
        "analyst":        AnalystAgent(model=model),
        "ac_coder":       CoderAgent(model=model, flow="analyst_coder_flow"),
        "coder_improver": CoderAgent(model=model, flow="tester_coder_flow"),
        "tester":         TesterAgent(model=model),
    }


# ─────────────────────────────────────────────
# Flow Dispatcher
# ─────────────────────────────────────────────

def dispatch_flow(flow_name: str, agents: dict, task_context: dict):
    """
    Instantiate and run the appropriate flow.

    Args:
        flow_name: One of the 6 configuration names.
        agents: Dict of agent instances from create_agents().
        task_context: Dict with keys: intent, method_name, test, OUTPUT_PATH,
                      task, task_id, provider, model_arg, api_key.
    """
    ctx = task_context  # shorthand

    if flow_name == "basic":
        flow = Flow(
            PYTHON_DEVELOPER=agents["basic_coder"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task=ctx["task"],
        )

    elif flow_name == "AC":
        flow = Analyst_Coder_Flow(
            PYTHON_DEVELOPER=agents["ac_coder"],
            ANALYST=agents["analyst"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task=ctx["task"],
        )

    elif flow_name == "ACT":
        flow = Analyst_Coder_Tester_Flow(
            CODER_MAIN=agents["ac_coder"],
            CODER_IMPROVER=agents["coder_improver"],
            ANALYST=agents["analyst"],
            TESTER=agents["tester"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task=ctx["task"],
        )

    elif flow_name == "debugger":
        flow = Debugger_Only_Flow(
            CODER_MAIN=agents["basic_coder"],
            CODER_IMPROVER=agents["coder_improver"],
            ANALYST=agents["analyst"],
            TESTER=agents["tester"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task_id=ctx["task_id"],
            provider=ctx["provider"],
            model=ctx["model_arg"],
            API_KEY=ctx["api_key"],
        )

    elif flow_name == "ac_debugger":
        flow = AC_Debug_Flow(
            CODER_MAIN=agents["basic_coder"],
            CODER_IMPROVER=agents["coder_improver"],
            ANALYST=agents["analyst"],
            TESTER=agents["tester"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task_id=ctx["task_id"],
            provider=ctx["provider"],
            model=ctx["model_arg"],
            API_KEY=ctx["api_key"],
        )

    elif flow_name == "act_debugger":
        flow = LDB_Flow(
            CODER_MAIN=agents["ac_coder"],
            CODER_IMPROVER=agents["coder_improver"],
            ANALYST=agents["analyst"],
            TESTER=agents["tester"],
            requirement=ctx["intent"],
            method_name=ctx["method_name"],
            test=ctx["test"],
            OUTPUT_PATH=ctx["OUTPUT_PATH"],
            task_id=ctx["task_id"],
            provider=ctx["provider"],
            model=ctx["model_arg"],
            API_KEY=ctx["api_key"],
        )

    else:
        raise ValueError(f"Unknown flow: {flow_name}")

    flow.run_flow()


# ─────────────────────────────────────────────
# Dataset Loading (HumanEval only)
# ─────────────────────────────────────────────

def load_humaneval(range_arg: str):
    """Load HumanEval dataset with the specified range."""
    dataset = load_dataset("openai_humaneval")

    if range_arg == 'full':
        dataset = DatasetDict({'test': dataset['test'].select(range(0, 164))})
    else:
        try:
            start, end = map(int, range_arg.split(':'))
            dataset = DatasetDict({'test': dataset['test'].select(range(start, end))})
        except ValueError:
            raise ValueError("Invalid range format. Use 'full' or 'start:end' (e.g., '0:10').")

    return dataset


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':

    OUTPUT_PATH = args.output_path
    FLOW = args.flow
    API_KEY = args.api_key

    # Parse provider and model
    try:
        provider, model_arg = args.provider_and_model.split(':')
    except ValueError:
        raise ValueError(
            "Invalid format. Use 'provider:model' "
            "(e.g., 'openai:gpt-4o-mini-2024-07-18')"
        )

    print(f"═══════════════════════════════════════════")
    print(f"  Model:  {provider}:{model_arg}")
    print(f"  Flow:   {FLOW}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"═══════════════════════════════════════════")

    # Initialize model (once for the entire run)
    model = create_model(provider, model_arg, API_KEY)

    # Initialize agents (once for the entire run)
    agents = create_agents(model)

    # Load dataset
    dataset = load_humaneval(args.range)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

    # Run experiment
    with open(OUTPUT_PATH, 'w+') as f:
        pass  # Create/truncate the file; flows append to it internally

    pbar = tqdm.tqdm(dataset['test'], total=len(dataset['test']))

    for idx, task in enumerate(pbar):

        method_name = task['entry_point']

        # Extract intent (full prompt including signature + docstring)
        _, _, _, _ = prompt_split_humaneval(task['prompt'], method_name)
        intent = task['prompt']  # Use full prompt as intent (signature mode)

        test = task['test']
        task_id = task['task_id']

        pbar.set_description(f"[{task_id}] {FLOW}")

        try:
            dispatch_flow(
                flow_name=FLOW,
                agents=agents,
                task_context={
                    "intent": intent,
                    "method_name": method_name,
                    "test": test,
                    "OUTPUT_PATH": OUTPUT_PATH,
                    "task": task,
                    "task_id": task_id,
                    "provider": provider,
                    "model_arg": model_arg,
                    "api_key": API_KEY,
                },
            )

        except RuntimeError as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            continue

    print(f"\n✅ Done. Results saved to: {OUTPUT_PATH}")
