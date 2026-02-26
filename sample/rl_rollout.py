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
from pathlib import Path
from termcolor import cprint
from jinja2 import Template
import torch.multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


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


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


# === Global cache for sequential sampling ===
_SHUFFLED_INDICES_CACHE = {}


def random_select(data_list, random_k):
    """Original random sampling function"""
    data_list = random.sample(data_list, random_k)
    return data_list


def get_sequential_batch(ordered_indices, global_cursor, batch_size, total_size):
    """
    Get next batch of indices sequentially with wrap-around.
    Returns: (batch_indices, new_cursor, completed_pass)
    """
    batch = []
    cursor = global_cursor
    for _ in range(batch_size):
        idx = ordered_indices[cursor % total_size]
        batch.append(idx)
        cursor += 1
    completed_pass = (cursor // total_size) > (global_cursor // total_size)
    return batch, cursor, completed_pass


def save_cursor(project_name, cursor):
    """Save cursor to file"""
    cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
    cursor_file.parent.mkdir(parents=True, exist_ok=True)
    cursor_file.write_text(str(cursor))


def load_cursor(project_name):
    """Load cursor from file; return None if not exists or unreadable."""
    cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
    if not cursor_file.exists():
        return None
    try:
        return int(cursor_file.read_text().strip())
    except (ValueError, OSError):
        return None


def to_single_token_stop_ids(tokenizer, stop_token_list):
    """Convert stop token list to single token IDs.
    
    Handles multiple input formats:
    - int: direct token ID
    - str: token string to encode
    - list/tuple of ints: sequence of token IDs
    """
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
        # Count tokens up to and including <|im_end|> (EOS for response)
        if "<|im_end|>" in s:
            s = s.split("<|im_end|>")[0] + "<|im_end|>"

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


def _kl_worker(gpu_id, data_chunk, policy_model_path, ref_model_path, 
               tokenizer_path, batch_size, max_length, result_queue):
    """
    Single GPU worker for KL computation.
    Each GPU loads both policy and ref models, processes its data chunk with batching.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    
    # Add project root to path for models import
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models import SDARForCausalLM
    
    device = f"cuda:{gpu_id}"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # Load policy model
        policy_model = SDARForCausalLM.from_pretrained(
            policy_model_path, trust_remote_code=True, torch_dtype="auto"
        )
        policy_model.eval().to(device)
        for param in policy_model.parameters():
            param.requires_grad = False
        
        # Load ref model
        ref_model = SDARForCausalLM.from_pretrained(
            ref_model_path, trust_remote_code=True, torch_dtype="auto"
        )
        ref_model.eval().to(device)
        for param in ref_model.parameters():
            param.requires_grad = False
        
        print(f"[KL Worker {gpu_id}] Models loaded, processing {len(data_chunk)} samples")
        
        # Collect all (sample_idx, resp_idx, prompt, response) items
        all_items = []
        for sample_idx, sample in data_chunk:
            prompt = sample["prompt"]
            responses = sample.get("full_output", [])
            for resp_idx, response in enumerate(responses):
                all_items.append((sample_idx, resp_idx, prompt, response))
        
        kl_results = {}
        
        # Process in batches
        from tqdm import tqdm
        num_batches = (len(all_items) + batch_size - 1) // batch_size
        
        for batch_start in tqdm(range(0, len(all_items), batch_size), 
                                 desc=f"GPU {gpu_id}", 
                                 disable=(gpu_id != 0),
                                 total=num_batches):
            batch = all_items[batch_start:batch_start + batch_size]
            
            # Prepare batch texts
            batch_texts = [item[2] + item[3] for item in batch]  # prompt + response
            
            # Tokenize batch with padding
            encodings = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            with torch.no_grad():
                # Policy model forward
                logits_policy = policy_model(input_ids, attention_mask=attention_mask).logits
                log_probs_policy = F.log_softmax(logits_policy, dim=-1)
                
                # Ref model forward
                logits_ref = ref_model(input_ids, attention_mask=attention_mask).logits
                log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                
                # Get logp for actual tokens (shifted by 1 for next-token prediction)
                labels = input_ids[:, 1:]
                logp_policy = log_probs_policy[:, :-1].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                logp_ref = log_probs_ref[:, :-1].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                
                # KL = logp_policy - logp_ref, masked by attention
                kl_batch = (logp_policy - logp_ref).cpu()
                mask_batch = attention_mask[:, 1:].cpu()
            
            # Store results for each item in batch
            for i, (sample_idx, resp_idx, _, _) in enumerate(batch):
                # Get valid length (non-padded tokens)
                valid_len = mask_batch[i].sum().item()
                kl_tensor = kl_batch[i, :int(valid_len)]
                
                if sample_idx not in kl_results:
                    kl_results[sample_idx] = {}
                # Convert to list for multiprocessing queue (avoid tensor serialization issues)
                kl_results[sample_idx][resp_idx] = kl_tensor.tolist()
        
        # Cleanup
        del policy_model, ref_model
        torch.cuda.empty_cache()
        
        print(f"[KL Worker {gpu_id}] Done, processed {len(all_items)} items")
        result_queue.put((gpu_id, kl_results))
        
    except Exception as e:
        import traceback
        print(f"[KL Worker {gpu_id}] Error: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, {}))


def compute_kl_penalty_in_rollout(config, data, tokenizer, project_name):
    """
    Compute KL penalty during rollout phase with multi-GPU and batch processing.
    
    Architecture:
    - Each GPU loads both policy and ref models
    - Data is split across GPUs
    - Each GPU processes its chunk with batching
    - Results are aggregated at the end
    
    KL_t = logp_policy - logp_ref
    """
    import torch
    import torch.multiprocessing as mp
    
    kl_in_reward_enabled = OmegaConf.select(config, "kl_in_reward.enabled", default=False)
    current_epoch = config.experiment.current_epoch
    
    # Epoch <= 1: policy == ref (SFT model), KL = 0, skip computation
    if not kl_in_reward_enabled:
        cprint("[KL] kl_in_reward not enabled, skipping", "yellow")
        return
    
    if current_epoch <= 1:
        cprint(f"[KL] Epoch {current_epoch} <= 1, policy == ref, KL = 0, skipping", "yellow")
        return
    
    # Get configuration
    num_gpus = torch.cuda.device_count()
    batch_size = OmegaConf.select(config, "training.batch_size_value", default=8)
    max_length = OmegaConf.select(config, "training.max_gen_length", default=8192)
    
    policy_model_path = os.path.abspath(project_name + "/ckpt/" + config.model.optimized_name)
    ref_model_path = config.model.pretrained_model
    tokenizer_path = policy_model_path  # Use policy model's tokenizer
    
    cprint(f"[KL] Multi-GPU KL computation: {num_gpus} GPUs, batch_size={batch_size}", "cyan")
    cprint(f"[KL] Policy model: {policy_model_path}", "cyan")
    cprint(f"[KL] Ref model: {ref_model_path}", "cyan")
    cprint(f"[KL] Total samples: {len(data)}", "cyan")
    
    # Prepare indexed data for distribution
    indexed_data = list(enumerate(data))
    
    # Split data across GPUs (interleaved for balance)
    data_chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(indexed_data):
        data_chunks[i % num_gpus].append(item)
    
    for gpu_id in range(num_gpus):
        cprint(f"[KL] GPU {gpu_id}: {len(data_chunks[gpu_id])} samples", "cyan")
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create result queue and spawn workers
    result_queue = mp.Queue()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=_kl_worker,
            args=(
                gpu_id,
                data_chunks[gpu_id],
                policy_model_path,
                ref_model_path,
                tokenizer_path,
                batch_size,
                max_length,
                result_queue
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results from all workers
    kl_results = {}
    for _ in range(num_gpus):
        gpu_id, partial_results = result_queue.get()
        cprint(f"[KL] Received results from GPU {gpu_id}: {len(partial_results)} samples", "cyan")
        for sample_idx, resp_dict in partial_results.items():
            if sample_idx not in kl_results:
                kl_results[sample_idx] = {}
            kl_results[sample_idx].update(resp_dict)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Convert lists back to tensors
    for sample_idx in kl_results:
        for resp_idx in kl_results[sample_idx]:
            kl_list = kl_results[sample_idx][resp_idx]
            kl_results[sample_idx][resp_idx] = torch.tensor(kl_list, dtype=torch.float32)
    
    # Save KL to file
    kl_file = os.path.abspath(project_name + "/temp_data/kl_penalty.pt")
    os.makedirs(os.path.dirname(kl_file), exist_ok=True)
    torch.save(kl_results, kl_file)
    cprint(f"[KL] Saved KL penalty to {kl_file} ({len(kl_results)} samples)", "green")


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    # Set multiprocessing start method
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Set CUDA architecture
    def _set_arch():
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    _set_arch()

    config = get_config()
    
    # === Prompt format support ===
    use_system_prompt = getattr(config.rollout, 'use_system_prompt', False)
    cprint(f"Prompt format: {'system+user+assistant' if use_system_prompt else 'user+assistant'}", "cyan")
    
    if use_system_prompt:
        system_prompts = '''<|im_start|>system\nPlease reason step by step, and put your final answer within $\\boxed{}$.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>system\nPlease reason step by step, and put your final answer within $\\boxed{}$.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>'''
    else:
        system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
        if config.rollout.start_with_think:
            system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant<think>\n'''

    project_name = config.experiment.project

    if config.experiment.current_epoch <= 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    code_task = False
    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task

        if config.dataset.data_type == "code":
            code_task = True
            if use_system_prompt:
                system_prompts_function = '''<|im_start|>system\nPlace your code within a single Python code block ```python ```. Do not include more than one code block.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
                system_prompts_stdio = '''<|im_start|>system\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
                if config.rollout.start_with_think:
                    system_prompts_function = '''<|im_start|>system\nPlace your code within a single Python code block ```python ```. Do not include more than one code block.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>'''
                    system_prompts_stdio = '''<|im_start|>system\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>'''
            else:
                system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
                system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
                if config.rollout.start_with_think:
                    system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''

        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        if config.evaluation.data_type == "code":
            code_task = True
            if use_system_prompt:
                system_prompts_function = '''<|im_start|>system\nPlace your code within a single Python code block ```python ```. Do not include more than one code block.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
                system_prompts_stdio = '''<|im_start|>system\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
                if config.rollout.start_with_think:
                    system_prompts_function = '''<|im_start|>system\nPlace your code within a single Python code block ```python ```. Do not include more than one code block.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>'''
                    system_prompts_stdio = '''<|im_start|>system\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script.<|im_end|>\n<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n<think>'''
            else:
                system_prompts_function = '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
                system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
                if config.rollout.start_with_think:
                    system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''

        k_sample = config.evaluation.num_response_per_task

        # Override rollout config with evaluation config
        config.rollout.tensor_parallel_size = config.evaluation.tensor_parallel_size
        config.rollout.max_active = config.evaluation.max_active
        config.rollout.max_token = config.evaluation.max_token
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.denoising_steps_per_block = config.evaluation.denoising_steps_per_block
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k
        config.rollout.block_size = config.evaluation.block_size
        if OmegaConf.select(config, "evaluation.cache_max_entry_count", default=None) is not None:
            config.rollout.cache_max_entry_count = config.evaluation.cache_max_entry_count
        # Override do_sample config for evaluation
        if OmegaConf.select(config, "evaluation.do_sample", default=None) is not None:
            config.rollout.do_sample = config.evaluation.do_sample

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # Load data
    with open("../data/" + dataset + ".json", 'r') as f:
        all_data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    
    # Get sequential sampling settings
    sequential_sampling = OmegaConf.select(config, "rollout.sequential_sampling", default=False)
    sampling_multiplier = OmegaConf.select(config, "rollout.sampling_multiplier", default=1)
    
    if config.experiment.function == "train":
        random_select_num = config.rollout.num_task_per_step
        random_select_num = int(random_select_num / num_node)
        random_select_num = int(random_select_num * sampling_multiplier)
        random_select_num = min(random_select_num, len(all_data))
        
        if sequential_sampling:
            cache_key = f"{dataset}"
            
            if cache_key not in _SHUFFLED_INDICES_CACHE:
                indices = list(range(len(all_data)))
                seed = OmegaConf.select(config, "training.seed", default=42)
                random.seed(seed)
                random.shuffle(indices)
                _SHUFFLED_INDICES_CACHE[cache_key] = indices
                cprint(f"[Sequential Sampling] Initialized shuffled indices for {dataset} (seed={seed})", "cyan")
            
            ordered_indices = _SHUFFLED_INDICES_CACHE[cache_key]

            # Prefer persisted cursor on disk; fallback to config default
            saved_cursor = load_cursor(project_name)
            if saved_cursor is not None:
                data_cursor = saved_cursor
            else:
                data_cursor = OmegaConf.select(config, "experiment.data_cursor", default=0)
            total_size = len(all_data)
            
            batch_indices, new_cursor, completed_pass = get_sequential_batch(
                ordered_indices, data_cursor, random_select_num, total_size
            )

            data = [all_data[i] for i in batch_indices]

            if completed_pass:
                # Check if in pretraining phase (no reshuffling during pretrain)
                is_pretraining_file = Path("../" + project_name) / "temp_data" / "is_pretraining.txt"
                if is_pretraining_file.exists():
                    cprint(f"[Sequential Sampling] Completed full dataset pass! (Pretraining: NO reshuffling)", "cyan")
                    # Reset cursor to 0 for next pass (keep same order)
                    cursor_file = Path("../" + project_name) / "temp_data" / "data_cursor.txt"
                    cursor_file.write_text("0")
                    cprint(f"[Sequential Sampling] Reset cursor to 0 (pretraining mode, keeping order)", "cyan")
                else:
                    cprint(f"[Sequential Sampling] Completed full dataset pass! Reshuffling...", "cyan")
                    random.shuffle(_SHUFFLED_INDICES_CACHE[cache_key])

                # Write completed_pass flag file (for value pretrain)
                pass_completed_file = Path("../" + project_name) / "temp_data" / "completed_pass.txt"
                pass_completed_file.parent.mkdir(parents=True, exist_ok=True)
                pass_completed_file.write_text(str(1))
                cprint(f"[Sequential Sampling] Wrote completed pass flag to {pass_completed_file}", "cyan")

            save_cursor(project_name, new_cursor)
            cprint(f"[Sequential Sampling] Cursor: {data_cursor} -> {new_cursor} (sampled {len(data)} tasks)", "cyan")
        else:
            if num_node > 1:
                random.shuffle(all_data)
                all_data = get_data_chunk(all_data, num_node, node_index)
            data = random_select(all_data, random_select_num)
    else:
        if num_node > 1:
            all_data = get_data_chunk(all_data, num_node, node_index)
        data = all_data
    
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
        if code_task:
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
        data[i]["generation_time"] = []
        data[i]["num_forwards"] = []
        data[i]["truncated"] = []
        data[i]["sample_idx"] = []  # Added: original prompt index for each response
        data[i]["resp_idx"] = []    # Added: index of each response within its prompt
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

    # Handle stop_token_list
    stop_token_ids = None
    if OmegaConf.select(config, "rollout.stop_token_list", default=MISSING) is not MISSING:
        stop_token_list = list(config.rollout.stop_token_list)
        stop_token_ids = to_single_token_stop_ids(tokenizer, stop_token_list)
        cprint(f"Using stop_token_list: {stop_token_list} -> token_ids: {stop_token_ids}", "cyan")
    elif use_system_prompt:
        default_stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        stop_token_ids = to_single_token_stop_ids(tokenizer, default_stop_tokens)
        cprint(f"Using default stop_token_list: {default_stop_tokens} -> token_ids: {stop_token_ids}", "cyan")

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

    # Read do_sample config, default to True for backward compatibility
    do_sample = OmegaConf.select(config, "rollout.do_sample", default=True)

    gen_kwargs = dict(
        do_sample=do_sample,
        temperature=config.rollout.temperature,
        top_k=config.rollout.top_k,
        top_p=config.rollout.top_p,
        max_new_tokens=config.rollout.max_token,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
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
        if code_task:
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

        # Detect truncation: response reached max_token length
        is_truncated = response_length[i] >= config.rollout.max_token
        data[index_i]["truncated"].append(is_truncated)

        # Calculate num_forwards from step_map
        if step_map_i and len(step_map_i) > 0:
            num_forwards = len(set(step_map_i))
        else:
            num_forwards = 0

        # Generation time placeholder
        generation_time_approx = total_generation_time / N if N > 0 else 0.0

        data[index_i]["generation_time"].append(generation_time_approx)
        data[index_i]["num_forwards"].append(num_forwards)

        # Record sample_idx and resp_idx for correct advantage grouping in downstream computation
        resp_idx = len(data[index_i]["full_output"]) - 1  # Index of current response within this prompt
        data[index_i]["sample_idx"].append(index_i)
        data[index_i]["resp_idx"].append(resp_idx)

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

    # Compute KL penalty during rollout (only for training, not evaluation)
    if config.experiment.function == "train" and node_index == 0:
        compute_kl_penalty_in_rollout(config, data, tokenizer, "../" + project_name)
