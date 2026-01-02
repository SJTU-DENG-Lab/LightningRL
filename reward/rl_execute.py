import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config():
    cli_conf   = OmegaConf.from_cli()
    yaml_conf  = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


from concurrent.futures import as_completed

import textwrap

def _run_many_pipe(snippet: str, tests: list[str], conn):
    import textwrap
    results = []
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
        conn.send(results)
    except SystemExit:
        conn.send([True] * len(tests))
    except Exception:
        conn.send([False] * len(tests))
    finally:
        try: conn.close()
        except Exception: pass


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int,
                        spawn_slack: float = 2.0) -> list[bool]:
    import time, multiprocessing as mp
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_many_pipe, args=(snippet, tests, child_conn), daemon=True)
    p.start()
    child_conn.close()

    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = remaining if remaining < 0.05 else 0.05
            if parent_conn.poll(wait):
                try:
                    res = parent_conn.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent_conn.poll(0.05):
                    try:
                        res = parent_conn.recv()
                    except EOFError:
                        res = None
                break

        if res is None and parent_conn.poll(0.05):
            try:
                res = parent_conn.recv()
            except EOFError:
                res = None

        if res is None:
            if p.is_alive():
                p.terminate()
            res = [False] * len(tests)
    finally:
        try: p.join(timeout=0.5)
        except Exception: pass
        try: parent_conn.close()
        except Exception: pass

    return [bool(x) for x in res]

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_function_dataset(data: list[dict], n_workers: int | None = None):
    import os
    n_cpu = os.cpu_count() or 4
    n_workers = max(1, int(n_workers)) if n_workers is not None else n_cpu

    for item in data:
        m_code = len(item["extracted_output"])
        m_test = len(item["test_list"])
        item["execution_result"] = [[None]  * m_test for _ in range(m_code)]
        item["correctness"]      = [[False] * m_test for _ in range(m_code)]
        item.setdefault("step_map", [])

    tasks = []
    for idx, item in enumerate(data):
        t_limit = item.get("test_time_limit", 1)
        tests   = item["test_list"]
        for i, snippet in enumerate(item["extracted_output"]):
            tasks.append((idx, i, snippet, tests, t_limit))

    futures = {}
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(max_workers=n_workers) as pool, \
        tqdm(total=len(tasks)*len(data[0]["test_list"]), desc=f"Function tests ({n_workers} threads)",
            dynamic_ncols=True, mininterval=0.1, miniters=1) as pbar:

        for idx, i, snippet, tests, t_limit in tasks:
            fut = pool.submit(_check_snippet_many, snippet, tests, t_limit)
            futures[fut] = (idx, i)

        for fut in as_completed(futures):
            idx, i = futures[fut]
            try:
                ok_list = fut.result()
            except Exception:
                ok_list = [False] * len(data[idx]["test_list"])

            for j, ok in enumerate(ok_list):
                data[idx]["execution_result"][i][j] = bool(ok)
                data[idx]["correctness"][i][j]      = bool(ok)
                pbar.update(1)

    return data




def worker_stdio(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin



def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results

def test_if_eq(x, y):  
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    size, rem = divmod(n, num_chunks)
    idx, start = [], 0
    for i in range(num_chunks):
        extra = 1 if i < rem else 0
        end   = start + size + extra
        idx.append((start, end)); start = end
    return idx







from tqdm import tqdm 

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list,
                           worker, num_chunks):
    chunks = get_chunk_indices(len(code_list), num_chunks)

    exe_results = []
    pbar = tqdm(total=len(code_list), desc=f"STDIO tests ({num_chunks} ch)")

    for start, end in chunks:
        sub_code_list       = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]

        sub_exe_results = run_scripts_with_timeout(
            sub_code_list,
            sub_test_input_list,
            sub_time_limit_list,
            worker
        )
        exe_results.extend(sub_exe_results)
        pbar.update(end - start)   

    pbar.close()             
    return exe_results


def evaluate_stdio_dataset(data: list[dict], num_chunks: int):
    
    idx_code, idx_case = [], []
    code_list, inp_list, tl_list = [], [], []

    for idx, item in enumerate(data):
        tl = item.get("test_time_limit", 1)
        m_code = len(item["extracted_output"])
        m_case = len(item["test_input"])

        data[idx]["execution_result"] = [[] for _ in range(m_code)]
        data[idx]["correctness"] = [[] for _ in range(m_code)]
        item.setdefault("step_map",           [])

        for c_idx, code in enumerate(item["extracted_output"]):
            for k in range(m_case):
                idx_code.append((idx, c_idx))  
                idx_case.append(k)      
                code_list.append(code)
                inp_list.append(item["test_input"][k])
                tl_list.append(tl)


    exe_results = run_scripts_with_chunk(
        code_list, inp_list, tl_list, worker_stdio, num_chunks
    )

    for i, res in enumerate(exe_results):
        idx, c_idx = idx_code[i]
        k          = idx_case[i]
        item       = data[idx]


        while len(item["execution_result"][c_idx]) < k + 1:
            item["execution_result"][c_idx].append("")
            item["correctness"][c_idx].append(False)
        item["execution_result"][c_idx][k] = res
        exp_out = item["test_output"][k]
        item["correctness"][c_idx][k]      = test_if_eq(res, exp_out)

    return data





def main():
    config          = get_config()
    project_name = config.experiment.project
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    if num_node > 1:
        file_name    = f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"
    else:
        file_name    = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    func_items  = [itm for itm in data if itm.get("test_method","function") == "function"]
    stdio_items = [itm for itm in data if itm.get("test_method") == "stdio"]

    # --- 1) function ---
    if func_items:
        updated_func = evaluate_function_dataset(func_items, n_workers=config.execute.num_chunk)
        func_iter = iter(updated_func)
        for i,it in enumerate(data):
            if it.get("test_method","function") == "function":
                data[i] = next(func_iter)


    # --- 2) stdio ---
    if stdio_items:
        total_scripts = sum(len(it["extracted_output"]) for it in stdio_items)
        num_chunks    = max(1, math.ceil(total_scripts / config.execute.num_chunk))
        updated_stdio = evaluate_stdio_dataset(stdio_items, num_chunks=num_chunks)
        it_stdio = iter(updated_stdio)
        for i, it in enumerate(data):
            if it.get("test_method") == "stdio":
                data[i] = next(it_stdio)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
