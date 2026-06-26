"""
Shared helpers for the three debugger flows:
  - flows/llm_debugger_flow.py
  - flows/debugger_only_flow.py
  - flows/AC_Debug_flow.py

Single source of truth for:
  - comment / code-preparation helpers
  - tracing helpers (divide, get_trace, instrument_simple_block, ...)
  - the LDB block-trace orchestrator (get_code_traces_block)
  - the IMPORT_HEADER and DEBUG module-level constants
"""

import os
import os.path
import random
import re
import time
import ast
from typing import List

from astroid import nodes
from astroid.builder import AstroidBuilder

from staticfg import CFGBuilder


IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

DEBUG = False


# --- comment / code-preparation helpers ----------------------------------------

def find_comment(func_impl: str, entry: str ) -> bool:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + "("):
            break
    func_body = "\n".join(func_impl_lines[i:])
    if func_body.find('\"\"\"') != -1 or func_body.find('\'\'\'') != -1:
        return True
    return False

def get_function(prompt):
    lines = prompt.split('\n')
    cur_func = ""
    funcs = []
    for i, l in enumerate(lines):
        if l.startswith("def "):
            if cur_func == "":
                cur_func = l
            else:
                funcs.append([func_name, cur_func])
                cur_func = l
            func_name = l.split("def ")[1].split("(")[0]
        elif cur_func != "":
            cur_func += "\n" + l
    return funcs

def fix_func_impl_comments(func_impl: str, prompt: str, entry) -> str:
    # extract comments from prompt and insert them into func_impl after the function header
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[1]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[1]
    # Get the function header
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def') and entry in line:
            break
    # Insert comments after the function header
    func_impl_lines.insert(i+1, '    \"\"\"' + comments + '\"\"\"')
    return '\n'.join(func_impl_lines)


def prepare_function_from_seed(dataset_type, prompt, seed, entry_point):
    if dataset_type in ["HumanEval", "MBPP"]:
        if (prompt in seed) or (('def ' + entry_point + '(') in seed):
            # It has the function header, no need to add
            cur_func_impl = seed
        else:
            cur_func_impl = prompt + "\n" + seed
        # Add auxilary function
        funcs = get_function(prompt)
        seed_funcs = [func[0] for func in get_function(seed)]
        for func in funcs:
            if func[0] not in seed_funcs:
                cur_func_impl = func[1] + "\n" + cur_func_impl
        # Add comments
        if not find_comment(cur_func_impl, entry_point):
            cur_func_impl = fix_func_impl_comments(cur_func_impl, prompt, entry_point)
    elif dataset_type in ["TransCoder"]:
        # It contains a whole program
        cur_func_impl = seed
    # Add import header
    if IMPORT_HEADER not in cur_func_impl:
        cur_func_impl = IMPORT_HEADER + cur_func_impl
    assert isinstance(cur_func_impl, str)
    return cur_func_impl


def insert_comment(func_impl: str, comment: str, entry: str) -> str:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + '('):
            break
    func_impl_lines.insert(i + 1, '    \"\"\"' + comment + '\"\"\"')
    return '\n'.join(func_impl_lines)

def remove_comment(old_block: List[str]) -> str:
    new_block = []
    old_block_lines = old_block.split('\n')
    for line in old_block_lines:
        if line.lstrip().startswith('#'):
            continue
        new_block.append(line)
    if len(new_block) == 1:
        return new_block[0]
    else:
        return '\n'.join(new_block)

def extrace_comment(prompt: str) -> str:
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[-2]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[-2]
    return comments


def add_comments(cur_func_impl, item):
    if not find_comment(cur_func_impl, item["entry_point"]):
        debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])
    else:
        debug_cur_func_impl = cur_func_impl

    return debug_cur_func_impl


# --- tracing helpers -----------------------------------------------------------

def divide(prog):
    try:
        cfg = CFGBuilder().build_from_src('block', prog)
    except Exception as e:
        return None, str(e)
    divided_block = []
    prog_lines = prog.split("\n")
    for block in cfg:
        divided_block.append([block, prog_lines[block.at():block.end()+1], block.id])
    return divided_block, None

def get_error_msg(error):
    error_lines = error.split('\n')
    error_msg = ""
    last_l = ""
    code = ""
    for l in error_lines:
        if "File \"" in last_l:
            code = l
        elif "Error: " in l:
            error_msg = ("This line is wrong: ```" + code + "```\n" + l) if "__var_list" not in code else l
            break
        last_l = l
    return error_msg

# Get trace
def get_trace_line(trace, funcname, fname):
    mark = f"--- modulename: .tmp.py, funcname: {funcname}" + "\n"
    lines = trace.split(mark)[1].split("\n")
    traces = []
    for l in lines:
        # trace also record comment lines for some reason
        if l.lstrip().startswith("\'\'\'") or l.lstrip().startswith("\"\"\"") or l.lstrip().startswith("#"):
            continue
        traces.append(l)
    return traces

# Return: "*timeout*" or "*execution fail*{error_msg}" or "*parse fail*{ferr}" or line_traces(List)
def get_trace(prog, funcname):
    fname = '.tmp.py.' + str(random.randint(0, 10000))
    f = open(fname, "w")
    f.write(prog)
    f.close()
    # run in command line python -m trace -t tmp.py > trace
    import subprocess
    try:
        res=subprocess.run(["python3", "-m", "trace", "-t", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except AssertionError:
        # This is expected if fail the test assetion
        pass
    except subprocess.TimeoutExpired:
        return "*timeout*"
    except Exception as e:
        error_msg = get_error_msg(res.stderr.decode('utf-8'))
        print("Trace Execution Fail:" + error_msg)
        return "*execution fail*" + error_msg
    finally:
        os.remove(fname)
    trace = res.stdout.decode('utf-8')
    # Find --- modulename: tmp, funcname: {funcname}
    try:
        trace = get_trace_line(trace, funcname, fname)
    except IndexError:
        ferr_name = "../error/.error.py" + str(time.time())
        if DEBUG:
            ferr = open(ferr_name, 'w')
            ferr.write(prog)
            ferr.close()
        return f"*parse fail*{ferr_name}"
    # Find all lines with .tmp.py
    line_trace = []
    for l in trace:
        if l.startswith(fname):
            import re
            m = re.search(f"^{fname}", l)
            if (not line_trace) or (line_trace[-1] not in l):
                line_trace.append(l[m.end():])
    return line_trace

def collect_runtime_value_simple(value_prof_prog):
    hook = ""
    import sys
    hooked_prog = hook + "\n" + value_prof_prog
    fname = "tmp_line.py" + f".{random.randint(0,10000)}"
    with open(fname, "w") as f:
        f.write(hooked_prog)
    import subprocess
    try:
        res=subprocess.run(["python3", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except subprocess.TimeoutExpired:
        return "*timeout*"
    finally:
        os.remove(fname)
    output = res.stderr.decode('utf-8')
    if "Traceback (most recent call last):" in output and ("AssertionError" not in output):
        output = get_error_msg(output)
        return "*execution fail*" + output
    output = res.stdout.decode('utf-8')
    return output

def get_lineno(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return int(trace_line[match.start()+1:match.end()-2])

def get_line(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return trace_line[match.end()+1:]

def get_indent(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    len1 = len(trace_line[match.end()+1:])
    len2 = len(trace_line[match.end()+1:].lstrip())
    return int((len1-len2)/4)

def extract_value(output):
    output = output.split("\n")[:-1]
    output = [x for x in output if x.startswith('Value_')]
    return output

def parse_runtime_value_simple_block(output, trace_lines):
    trace_idx = 0
    blocks = []
    blk = []
    value_profiles = extract_value(output)
    trace_len = len(trace_lines)
    trace_linenos = [get_lineno(l) for l in trace_lines]
    last_bp = ""
    trace_idx = 0
    for i, l in enumerate(value_profiles):
        if trace_idx >= trace_len:
            break
        lineno = int(l.split(':')[1].split('|')[0])
        values = '\t'.join(l.split('|')[1:])
        values = values if len(values) < 100 else (values[:50] + "..." + values[-50:])
        if lineno not in trace_linenos:
            #payload = "    "*get_indent(trace_lines[trace_idx]) + "# " + values
            last_bp = values
            continue
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp)
        while trace_idx < trace_len and get_lineno(trace_lines[trace_idx]) != lineno:
            trace_l = trace_lines[trace_idx]
            blk.append(get_line(trace_l))
            trace_idx += 1
        if trace_idx == trace_len:
            break
        blk.append(get_line(trace_lines[trace_idx]))
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + values)
        last_bp = values
        blocks.append(blk)
        blk = []
        trace_idx += 1
    if trace_idx < trace_len:
        blk = ["    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp] + blk
        while trace_idx < trace_len:
            blk.append(get_line(trace_lines[trace_idx]))
            trace_idx += 1
        blocks.append(blk)
    return blocks


def get_range(prog, entry):
    tree = AstroidBuilder().string_build(prog)
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) and ele.name == entry:
            return [ele.lineno-1, ele.end_lineno-1] # Lineno start from 0
    return None

def get_after(stmts):
    for s in stmts:
        if s == "":
            continue
        else:
            return s.strip(), int((len(s) - len(s.lstrip()))/4)


def instrument_simple_block(prog, entry, divided_blocks):
    stmts = prog.split("\n")
    # Get range of entry function
    rang = get_range(prog, entry)
    block_insert = set([b[0].at() - 1 for b in divided_blocks] + [b[0].end() for b in divided_blocks])
    if rang is None:
        assert False, f"{entry} not in {prog}!"
    res = []
    for i, stmt in enumerate(stmts):
        if i < rang[0]:
            res.append(stmt)
            continue
        elif i > rang[1]:
            res.append(stmt)
            break
        if (i+1) not in block_insert:
            res.append(stmt)
            continue
        # indent the same as this statement
        refs, indent_after = get_after(reversed(stmts[:i+1]))
        # Unless
        if refs.startswith("else:") or refs.startswith("elif ") or refs.startswith("if ") or refs.startswith("while ") or refs.startswith("for ") or refs.startswith("def "):
            refs, indent_after = get_after(stmts[i+1:])
        payload = "    "*indent_after + f"__var_list = vars();print(f'Value_After:{i+1}|' + '|'.join([(x + '=' + _str(__var_list[x])) for x in __var_list if not x.startswith('__')]));"
        if stmt.find(" return ") != -1:
            stmt = stmt.replace(" return ", " _ret = ")
            payload = payload + " return _ret"
        res.append(stmt)
        res.append(payload)
    return "\n".join(res)


# --- LDB block-trace orchestrator ----------------------------------------------

def get_code_traces_block(prog, test, entry):
    log_of_tracing = ""
    # Divide program into basic block units
    divided_blocks, error = divide(prog)
    prog_lines = prog.split("\n")

    if divided_blocks is None:
        return "*execution fail*" + error
    # Collect Execution Traces
    if test.find("assert ") != -1:
        try:
            parsed = ast.parse(test)
            try:
                call_node = parsed.body[0].test.left
            except AttributeError:
                call_node = parsed.body[0].test
            test = f"print({ast.unparse(call_node)})"
        except SyntaxError:
            pass
    exec_prog = prog + "\n" + test
    trace_lines = get_trace(exec_prog, entry)
    if isinstance(trace_lines, str):
        if trace_lines == "*timeout*" or trace_lines.startswith("*execution fail*") or trace_lines.startswith("*parse fail*"):
            return trace_lines
    log_of_tracing += str("Trace:\n"+ '\n'.join(trace_lines[:10]))
    value_prof_prog = instrument_simple_block(prog, entry, divided_blocks)
    log_of_tracing += str("\nValue Profile Program:\n" + value_prof_prog + "\n" + test + "\n")
    output = collect_runtime_value_simple(value_prof_prog + "\n" + test)
    if output == "*timeout*" or output.startswith("*execution fail*"):
        return output
    log_of_tracing += "\n" + str("Value Profile Output:\n" + output)
    runtime_value = parse_runtime_value_simple_block(output, trace_lines)

    if not os.path.exists("./tracing_log"):
        os.makedirs("./tracing_log")
    log_file = "./tracing_log/trace.log."+str(random.randint(0, 10000))
    with open(log_file, 'w') as f:
        f.write(log_of_tracing)
        print(f"Writing tracing logs to {log_file}")
    return runtime_value
