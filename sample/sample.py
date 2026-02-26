import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Read GPU configuration early (will be overridden by config later if specified)
import sys
if '--config' in ' '.join(sys.argv) or 'config=' in ' '.join(sys.argv):
    from omegaconf import OmegaConf
    for arg in sys.argv:
        if arg.startswith('config='):
            config_path = arg.split('=', 1)[1]
            try:
                temp_conf = OmegaConf.load(config_path)
                if hasattr(temp_conf.experiment, 'gpu_ids') and temp_conf.experiment.gpu_ids:
                    _os.environ["CUDA_VISIBLE_DEVICES"] = str(temp_conf.experiment.gpu_ids)
                    print(f"Setting CUDA_VISIBLE_DEVICES={temp_conf.experiment.gpu_ids}")
            except:
                pass
            break


# Consolidate all caches into the local high-speed disk (NVMe or /dev/shm)
_cache_root = "/dev/shm/torch_cache"
_os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"] = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"] = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import os
import re
import json
import gc
import time
import math
import random
import queue
from termcolor import cprint
from jinja2 import Template
import torch.multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


# MATH 4-shot examples from "Many-Shot In-Context Learning" paper (arXiv 2404.11018, Figure A.6)
MATH_4SHOT_EXAMPLES = '''<|im_start|>user
Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$<|im_end|>
<|im_start|>assistant
I need to find the domain of this expression.
This expression has two square root expressions.
I know the values under each square root must be non-negative.
So let me start by ensuring that.
I set $x-2 \\geq 0$, which simplifies to $x \\geq 2$.
Then, I set $5-x \\geq 0$, which simplifies to $x \\leq 5$.
But I need to make sure the denominator is not zero.
So, $5-x$ must also be greater than zero, which gives me $x < 5$.
Therefore, the domain of the expression is $[2,5)$ $\\boxed{[2,5)}$
Answer: $[2,5)$
Final Answer: The final answer is $[2,5)$. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: If $\\det\\mathbf{A}=2$ and $\\det\\mathbf{B}=12$, then find $\\det(\\mathbf{A}\\mathbf{B})$.<|im_end|>
<|im_start|>assistant
I need to find the determinant of the product of matrices A and B.
I remember that the determinant of the product of two matrices equals the product of their determinants.
So, $\\det(\\mathbf{A}\\mathbf{B})=(\\det\\mathbf{A})(\\det\\mathbf{B})=(2)(12)=\\boxed{24}$
Answer: 24
Final Answer: The final answer is 24. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?<|im_end|>
<|im_start|>assistant
Okay, so Terrell lifts a total of $2 \\cdot 12 \\cdot 20 = 480$ pounds of weight with the 20-pound weights.
Well, if he switches to 15-pound weights, the total weight lifted will be $2 \\cdot 15 \\cdot n = 30n$ pounds, where n is the number of lifts.
I want to find the number of lifts, n, for the total weight lifted to be the same.
I equate $30n$ to 480 pounds and solve for n.
$30n = 480$
$\\Rightarrow n = 480/30 = \\boxed{16}$
Answer: 16
Final Answer: The final answer is 16. I hope it is correct.<|im_end|>
<|im_start|>user
Problem: If the system of equations $6x-4y=a$, $6y-9x=b$ has a solution $(x,y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b}$, assuming $b$ is nonzero.<|im_end|>
<|im_start|>assistant
I'm given a system of two equations.
I see that if I multiply the first equation by $-\\frac{3}{2}$, I'll get another equation that has the same left-hand side as the second equation, $6y-9x$.
Let me try that
$6y-9x=-\\frac{3}{2}a$.
Ah, I also know that $6y-9x=b$, so I can equate these two equations.
So, $-\\frac{3}{2}a=b \\Rightarrow \\frac{a}{b}=\\boxed{-\\frac{2}{3}}$.
Answer: $-\\frac{2}{3}$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.<|im_end|>
<|im_start|>user
'''


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output):
    """Extract code from model output using OpenCompass multi-pattern matching logic

    Args:
        full_output: Full text output from model

    Returns:
        Extracted code string, or cleaned version of raw output if extraction fails
    """
    # OpenCompass 17-pattern matching order
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)'\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*DONE",
        r"BEGIN\s*'(.*)'\s*DONE",
        r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
        r'BEGIN\s*(.*)\s*\[DONE\]',
        r'\[BEGIN\]\s*(.*)\s*DONE',
        r'BEGIN\s*(.*)\s*DONE',
        r'```python\s*(.*)\s*```',
        r'```\s*(.*)\s*```',
        r'```python\s*(.*)\s*$',
        r'```\s*(.*)\s*$',
        r'(.*)\s*```.*',
        r"\[BEGIN\]\s*'(.*)",
        r'\[BEGIN\](.*)',
        r"'(.*)'\s*\[DONE\]",
    ]

    # Try patterns in order
    for p in patterns:
        try:
            match = re.search(p, full_output, re.DOTALL)
        except:
            match = None

        if match:
            full_output = match.group(1)
            break

    # Fallback 1: split on ``` and take first part
    full_output = full_output.split('```')[0]

    # Fallback 2: split on [DONE] and take first part
    full_output = re.split(r"'?\s*\[?DONE\]?", full_output)[0]

    # Clean up
    full_output = full_output.replace('\\_', '_')
    full_output = full_output.strip()

    return full_output


def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


def to_single_token_stop_ids(tokenizer, stop_token_list):
    """Convert stop token list to single token IDs."""
    if not stop_token_list:
        return []
    ids, seen = [], set()
    for s in stop_token_list:
        if isinstance(s, int):
            tid = [s]
        elif isinstance(s, str):
            tid = tokenizer.encode(s, add_special_tokens=False)
        elif isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
            tid = list(s)
        else:
            continue
        if len(tid) == 1:
            t = tid[0]
            if t not in seen:
                seen.add(t)
                ids.append(t)
    return ids


def get_token_lengths(strings, tokenizer):
    """Calculate token lengths for a list of strings."""
    pad_token = tokenizer.pad_token

    escaped = re.escape(pad_token)
    pattern = rf"(?:{escaped})+"
    remove_pattern = escaped

    collapse_re = re.compile(pattern)

    lengths = []
    for s in strings:
        # Only count tokens up to <|im_end|> (EOS for response)
        if "<|im_end|>" in s:
            s = s.split("<|im_end|>")[0]

        s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
        s_clean = re.sub(remove_pattern, '', s_clean)
        lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
    return lengths


def _lmdeploy_worker_run(args):
    """LMDeploy worker for data parallelism."""
    (model_path, tp, backend_kwargs, gen_kwargs,
     vis_ids, prompts_slice, indices_slice, max_active) = args

    import os
    import time
    import gc

    # Set visible GPUs for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[worker pid={os.getpid()}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, "
          f"prompts={len(prompts_slice)}", flush=True)

    # Import after setting CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.set_device(0)

    # Add lmdeploy to path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

    # Build configs
    backend_config = PytorchEngineConfig(
        tp=tp,
        max_batch_size=min(max_active, max(1, len(prompts_slice))),
        **backend_kwargs
    )
    gen_config = GenerationConfig(**gen_kwargs)

    triples = []
    worker_generation_time = 0.0

    try:
        start_time = time.perf_counter()

        with pipeline(model_path, backend_config=backend_config) as pipe:
            outputs = pipe(prompts_slice, gen_config=gen_config,
                          do_preprocess=False, use_tqdm=True)

            for j, o in enumerate(outputs):
                triples.append((
                    indices_slice[j],
                    o.text,
                    o.step_map
                ))

        worker_generation_time = time.perf_counter() - start_time

    except Exception as e:
        print(f"[worker pid={os.getpid()}] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return triples, worker_generation_time


def _lmdeploy_worker_entry(args, out_q):
    """Entry point with error handling."""
    import traceback
    import os
    try:
        res, worker_time = _lmdeploy_worker_run(args)
        out_q.put(("ok", (res, worker_time)))
    except Exception:
        out_q.put(("err", {
            "pid": os.getpid(),
            "traceback": traceback.format_exc(),
        }))


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    # Set multiprocessing start method
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    config = get_config()

    # Set CUDA architecture
    def _set_arch():
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    _set_arch()

    k_sample = config.rollout.num_response_per_task

    # Default prompt template
    # Check if n-shot is enabled (only applies to MATH500 + math data type)
    n_shot = config.dataset.get("n_shot", 0)
    use_4shot = (n_shot == 4 and
                 config.dataset.eval_dataset == "MATH500" and
                 config.dataset.data_type == "math")

    if use_4shot:
        # Use 4-shot prompt
        system_prompts = MATH_4SHOT_EXAMPLES + '''{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
    elif config.rollout.start_with_think:
        system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant

\n'''
    else:
        # Default 0-shot prompt (n_shot=0 or other datasets)
        system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''

    project_name = config.experiment.project

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model

    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''
    elif config.dataset.data_type == "option":
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant<think>\n'''

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)

    num = len(data)

    model_path = os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    block_size = config.rollout.block_size
    tp = int(config.rollout.tensor_parallel_size)

    # Prepare prompts
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample

        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["num_forwards"] = []
        data[i]["prompt"] = get_prompt(data[i])

    # Shuffle prompts
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    # Determine GPU configuration
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[preflight] torch.cuda.device_count()={torch.cuda.device_count()}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        device_ids = [int(x) for x in visible_gpus]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    gpu_num = len(device_ids)
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."
    assert gpu_num >= 1, "No GPU visible"

    # Determine parallelism strategy
    if tp > 1:
        ngroups = 1  # Tensor Parallelism: single group using multiple GPUs
    else:
        ngroups = gpu_num  # Data Parallelism: one group per GPU

    groups = [device_ids[i*tp : (i+1)*tp] for i in range(ngroups)]

    print(f"[preflight] Total prompts: {N}, tensor_parallel_size: {tp}, ngroups: {ngroups}")

    # Get stop token IDs
    if OmegaConf.select(config, "rollout.stop_token_list", default=MISSING) is not MISSING:
        stop_token_id_list = to_single_token_stop_ids(tokenizer, config.rollout.stop_token_list)
    else:
        stop_token_id_list = []

    # Build config kwargs for workers
    cache_max_entry_count = OmegaConf.select(config, "rollout.cache_max_entry_count", default=0.8)
    backend_kwargs = dict(
        dllm_block_length=config.rollout.block_size,
        dllm_denoising_steps=config.rollout.denoising_steps_per_block,
        dllm_unmasking_strategy=config.rollout.remasking_strategy,
        dllm_confidence_threshold=config.rollout.dynamic_threshold,
        cache_max_entry_count=cache_max_entry_count,
        eager_mode=(tp == 1),
    )

    # Read do_sample config, default to False for evaluation (greedy decoding)
    do_sample = OmegaConf.select(config, "rollout.do_sample", default=False)

    gen_kwargs = dict(
        do_sample=do_sample,
        temperature=config.rollout.temperature,
        top_k=config.rollout.top_k,
        top_p=config.rollout.top_p,
        max_new_tokens=config.rollout.max_token,
        stop_token_ids=stop_token_id_list if stop_token_id_list else None,
    )

    max_active_local = config.rollout.max_active

    # Split prompts into chunks for each group
    def _chunk_by_groups(lst, ng):
        L = len(lst)
        if ng <= 1:
            return [lst]
        chunk_size = math.ceil(L / ng)
        return [lst[i*chunk_size : min((i+1)*chunk_size, L)] for i in range(ng)]

    prompt_chunks = _chunk_by_groups(shuffled_prompts, ngroups)
    index_chunks = _chunk_by_groups(shuffled_idx, ngroups)

    for a, b in zip(prompt_chunks, index_chunks):
        assert len(a) == len(b)

    seq_pairs = []
    total_generation_time = 0.0

    if ngroups == 1:
        # Single group: run in main process (TP mode or single GPU)
        # Add lmdeploy to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, groups[0]))
        torch.cuda.set_device(0)

        backend_config = PytorchEngineConfig(
            tp=tp,
            max_batch_size=max_active_local,
            **backend_kwargs
        )
        gen_config = GenerationConfig(**gen_kwargs)

        start_time = time.perf_counter()

        with pipeline(model_path, backend_config=backend_config) as pipe:
            outputs = pipe(prompt_chunks[0], gen_config=gen_config,
                          do_preprocess=False, use_tqdm=True)

            for j, o in enumerate(outputs):
                seq_pairs.append((
                    index_chunks[0][j],
                    o.text,
                    o.step_map
                ))

        total_generation_time = time.perf_counter() - start_time

        gc.collect()
        torch.cuda.empty_cache()

    else:
        # Multiple groups: spawn worker processes (DP mode)
        cprint(f"[Data Parallelism] Spawning {ngroups} workers...", "cyan")

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = []

        for g in range(ngroups):
            if len(prompt_chunks[g]) == 0:
                continue

            args = (
                model_path, tp, backend_kwargs, gen_kwargs,
                groups[g], prompt_chunks[g], index_chunks[g], max_active_local
            )
            p = ctx.Process(target=_lmdeploy_worker_entry, args=(args, out_q), daemon=False)
            p.start()
            procs.append(p)

        # Collect results from workers
        results_needed = len(procs)
        results_got = 0
        worker_times = []

        while results_got < results_needed:
            try:
                kind, payload = out_q.get(timeout=3600)  # 1 hour timeout
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    for p in dead:
                        print(f"[parent] worker pid={p.pid} exitcode={p.exitcode} (no result)", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    for p in procs:
                        p.join(timeout=5)
                    raise RuntimeError("Some workers died without returning results.")
                continue

            if kind == "ok":
                worker_results, worker_time = payload
                seq_pairs.extend(worker_results)
                worker_times.append(worker_time)
                results_got += 1
            else:  # "err"
                print(f"[parent] worker error:\n{payload['traceback']}", flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError("Worker failed. See traceback above.")

        for p in procs:
            p.join()

        # Total time is max of worker times (they run in parallel)
        total_generation_time = max(worker_times) if worker_times else 0.0
        total_gpu_time = sum(worker_times)
        print(f"[Data Parallelism] {ngroups} workers, wall time: {total_generation_time:.2f}s, "
              f"total GPU time: {total_gpu_time:.2f}s")

    # Restore original order
    restored_outputs = [None] * N
    restored_steps = [None] * N

    for item in seq_pairs:
        if len(item) == 2:
            gi, text = item
            steps = None
        else:
            gi, text, steps = item
        restored_outputs[gi] = text
        restored_steps[gi] = steps

    for i in range(N):
        if restored_outputs[i] is None:
            restored_outputs[i] = ""
        if restored_steps[i] is None:
            restored_steps[i] = []

    cprint(f"generation job done! Total time: {total_generation_time:.2f}s", "green")

    # Calculate response lengths
    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length) if response_length else 0

    # Process outputs
    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i / k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)

        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        step_map_i = restored_steps[i] if restored_steps[i] is not None else []
        data[index_i]["step_map"].append(step_map_i)
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])

        # Calculate num_forwards from step_map
        if step_map_i and len(step_map_i) > 0:
            num_forwards = len(set(step_map_i))
        else:
            num_forwards = 0

        data[index_i]["num_forwards"].append(num_forwards)
        i += 1

    # Save output
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    output_data = {
        "metadata": {
            "total_generation_time": total_generation_time,
            "num_samples": len(data),
            "num_gpu_workers": ngroups,
            "tensor_parallel_size": tp
        },
        "data": data
    }

    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    cprint(f"Results saved to {output_file_name}", "green")
