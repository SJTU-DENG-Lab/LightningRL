"""
train_policy_no_value.py

Responsible for data processing and advantage computation when no value model is used.
Loads rollout data from JSON, processes it, and saves as PT file for policy training.
Directly uses per-sequence rewards as advantage.
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import torch
import numpy as np
from pathlib import Path
from termcolor import cprint
from omegaconf import OmegaConf
from train.utils import get_config
from transformers import AutoTokenizer
from train.prompting_utils import UniversalPrompting

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_advantages_no_value(
    Return_mat, adv_mat, per_seq_reward, seq_ids, p_mask, L0,
    project_name, current_epoch, input_ids_lm=None, eos_id=None, L1=None,
    num_responses_per_task=None,
    sample_idx_all=None, resp_idx_all=None,
    adv_norm_mode="batch"  # advantage normalization mode: "batch", "prompt_group", "prompt_group_center"
):
    """
    Advantage computation when no value model is used.

    Normalization strategy is controlled by adv_norm_mode:
    - "batch": Batch-level normalization (reward - batch_mean) / batch_std
    - "prompt_group": Prompt group-level normalization (reward - group_mean) / group_std
    - "prompt_group_center": Prompt group centering (reward - group_mean), without dividing by std

    Notes:
    - speed reward (TPF) normalization is controlled by tpf_norm config (handled in rl_reward.py)
    - p_mask includes the prompt portion; advantage filling must exclude it

    Args:
        Return_mat: (B, L) return matrix
        adv_mat: (B, L) advantage matrix
        per_seq_reward: (B,) reward for each sequence
        seq_ids: (B,) sequence IDs
        p_mask: (B, L) mask (includes prompt portion)
        L0: prompt length
        project_name: project name
        current_epoch: current epoch
        input_ids_lm: (B, L) input token IDs (for EOS detection)
        eos_id: EOS token ID (for EOS special handling)
        L1: response length
        num_responses_per_task: number of responses per prompt (deprecated, use sample_idx grouping)
        sample_idx_all: (B,) original prompt index for each sample (for correct grouping)
        resp_idx_all: (B,) response index within the prompt for each sample
        adv_norm_mode: advantage normalization mode, "batch" or "prompt_group"
    """
    B = seq_ids.size(0)
    L = p_mask.size(1)

    # If not specified, try to get from config
    if num_responses_per_task is None:
        try:
            config = get_config()
            num_responses_per_task = config.rollout.num_response_per_task
        except:
            num_responses_per_task = 1  # Default: each response is independent

    logger.info(f"[No Value Model] Computing advantages for {B} sequences")
    logger.info(f"[No Value Model] Advantage normalization mode = {adv_norm_mode}")

    # Prevent division by zero when std=0
    eps = 1e-8

    # Select normalization strategy based on adv_norm_mode
    if adv_norm_mode == "batch":
        # ===== Batch-level normalization =====
        logger.info("[No Value Model] Using batch-level advantage normalization")

        # Collect all rewards across the entire batch
        all_rewards = per_seq_reward  # (B,) tensor
        batch_mean = all_rewards.mean().item()
        batch_std = all_rewards.std().item()
        batch_std = max(batch_std, eps)

        logger.info(f"[No Value Model] Batch stats: mean={batch_mean:.4f}, std={batch_std:.4f}")

        for s in range(B):
            seq_reward = per_seq_reward[s].item()
            normalized_adv = (seq_reward - batch_mean) / batch_std

            # Fill advantage only at response positions (excluding prompt)
            pm_resp = p_mask[s].clone()
            pm_resp[:L0] = False  # Exclude prompt portion

            adv_mat[s][pm_resp] = normalized_adv
            Return_mat[s][pm_resp] = seq_reward

        # Define num_prompts for saving statistics
        num_prompts = B // num_responses_per_task

    elif adv_norm_mode == "prompt_group":
        # ===== Prompt group-level normalization =====
        logger.info("[No Value Model] Using prompt-group-level advantage normalization")

        # Group by sample_idx for normalization (approach C: correct grouping via sample_idx)
        # Samples with the same sample_idx_all belong to the same prompt group
        # This correctly handles misaligned grouping after filtered/truncated samples
        if sample_idx_all is not None and resp_idx_all is not None:
            # Group by sample_idx
            prompt_groups = {}
            for s in range(B):
                sid = sample_idx_all[s].item() if isinstance(sample_idx_all, torch.Tensor) else sample_idx_all[s]
                prompt_groups.setdefault(sid, []).append(s)

            num_groups = len(prompt_groups)
            num_prompts = num_groups  # Used for saving statistics
            logger.info(f"[No Value Model] Using sample_idx-based grouping: {num_groups} prompt groups")

            for sid, indices in prompt_groups.items():
                # Extract rewards within group (using tensor indexing)
                group_rewards = per_seq_reward[indices]  # (len(indices),) tensor
                group_mean = group_rewards.mean().item()
                group_std = group_rewards.std().item()
                group_std = max(group_std, eps)

                # For each sequence in this prompt group, apply normalization: (reward - mean) / std
                for s in indices:
                    seq_reward = per_seq_reward[s].item()
                    normalized_adv = (seq_reward - group_mean) / group_std

                    # Fill advantage only at response positions (excluding prompt)
                    pm_resp = p_mask[s].clone()
                    pm_resp[:L0] = False  # Exclude prompt portion

                    adv_mat[s][pm_resp] = normalized_adv
                    Return_mat[s][pm_resp] = seq_reward
        else:
            # Fall back to fixed-size grouping (backward compatible, but not recommended for filtered data)
            num_prompts = B // num_responses_per_task
            logger.warning(f"[No Value Model] sample_idx not available, using fixed-size grouping: {num_prompts} prompts, {num_responses_per_task} responses per prompt")

            for prompt_idx in range(num_prompts):
                start_seq = prompt_idx * num_responses_per_task
                end_seq = start_seq + num_responses_per_task

                # Extract rewards within group (using tensor indexing)
                group_rewards = per_seq_reward[start_seq:end_seq]
                group_mean = group_rewards.mean().item()
                group_std = group_rewards.std().item()
                group_std = max(group_std, eps)

                # For each sequence in this prompt group, apply normalization: (reward - mean) / std
                for s in range(start_seq, end_seq):
                    seq_reward = per_seq_reward[s].item()
                    normalized_adv = (seq_reward - group_mean) / group_std

                    # Fill advantage only at response positions (excluding prompt)
                    pm_resp = p_mask[s].clone()
                    pm_resp[:L0] = False  # Exclude prompt portion

                    adv_mat[s][pm_resp] = normalized_adv
                    Return_mat[s][pm_resp] = seq_reward

    elif adv_norm_mode == "prompt_group_center":
        # ===== Prompt group centering (without dividing by std) =====
        logger.info("[No Value Model] Using prompt-group-level centering (no std division)")

        if sample_idx_all is not None and resp_idx_all is not None:
            # Group by sample_idx
            prompt_groups = {}
            for s in range(B):
                sid = sample_idx_all[s].item() if isinstance(sample_idx_all, torch.Tensor) else sample_idx_all[s]
                prompt_groups.setdefault(sid, []).append(s)

            num_groups = len(prompt_groups)
            num_prompts = num_groups
            logger.info(f"[No Value Model] Using sample_idx-based grouping: {num_groups} prompt groups")

            for sid, indices in prompt_groups.items():
                # Extract rewards within group
                group_rewards = per_seq_reward[indices]
                group_mean = group_rewards.mean().item()

                # Centering: subtract mean without dividing by std
                for s in indices:
                    seq_reward = per_seq_reward[s].item()
                    centered_adv = seq_reward - group_mean

                    # Fill advantage only at response positions
                    pm_resp = p_mask[s].clone()
                    pm_resp[:L0] = False

                    adv_mat[s][pm_resp] = centered_adv
                    Return_mat[s][pm_resp] = seq_reward
        else:
            # Fall back to fixed-size grouping
            num_prompts = B // num_responses_per_task
            logger.warning(f"[No Value Model] sample_idx not available, using fixed-size grouping: {num_prompts} prompts")

            for prompt_idx in range(num_prompts):
                start_seq = prompt_idx * num_responses_per_task
                end_seq = start_seq + num_responses_per_task

                group_rewards = per_seq_reward[start_seq:end_seq]
                group_mean = group_rewards.mean().item()

                for s in range(start_seq, end_seq):
                    seq_reward = per_seq_reward[s].item()
                    centered_adv = seq_reward - group_mean

                    pm_resp = p_mask[s].clone()
                    pm_resp[:L0] = False

                    adv_mat[s][pm_resp] = centered_adv
                    Return_mat[s][pm_resp] = seq_reward

    else:
        raise ValueError(f"Invalid adv_norm_mode: {adv_norm_mode}. Must be 'batch', 'prompt_group', or 'prompt_group_center'")

    # Statistics
    num_nonzero_adv = (adv_mat != 0).sum().item()
    mean_adv = adv_mat[adv_mat != 0].mean().item() if num_nonzero_adv > 0 else 0.0
    std_adv = adv_mat[adv_mat != 0].std().item() if num_nonzero_adv > 0 else 0.0

    # Compute Collapse_Ratio (GPU-accelerated + sampling)
    non_zero_adv = adv_mat[adv_mat != 0].flatten()  # Keep as torch tensor on original device (GPU)
    if len(non_zero_adv) > 1:
        adv_std = non_zero_adv.std()

        # Sampling: at most 150000 points (approx. 84GB VRAM)
        if len(non_zero_adv) > 150000:
            indices = torch.randperm(len(non_zero_adv), device=non_zero_adv.device)[:150000]
            sample_adv = non_zero_adv[indices]
        else:
            sample_adv = non_zero_adv

        # Compute difference matrix on GPU
        diff_matrix = torch.abs(sample_adv[:, None] - sample_adv[None, :])
        # Take upper triangle (excluding diagonal)
        triu_mask = torch.triu(torch.ones_like(diff_matrix), diagonal=1).bool()
        upper_tri = diff_matrix[triu_mask]

        collapse_ratios = {}
        for threshold in [0.01, 0.05, 0.1, 0.5]:
            close_pairs = (upper_tri < threshold).sum().item()
            total_pairs = len(upper_tri)
            collapse_ratios[threshold] = close_pairs / total_pairs
    else:
        collapse_ratios = {0.01: 0.0, 0.05: 0.0, 0.1: 0.0, 0.5: 0.0}

    logger.info(f"[No Value Model] Filled {num_nonzero_adv} advantage values")
    logger.info(f"[No Value Model] Advantage statistics: mean={mean_adv:.4f}, std={std_adv:.4f}")
    logger.info(f"[No Value Model] Collapse Ratios: 0.01={collapse_ratios[0.01]:.4f}, 0.05={collapse_ratios[0.05]:.4f}, 0.1={collapse_ratios[0.1]:.4f}, 0.5={collapse_ratios[0.5]:.4f}")

    # Save statistics to file
    stats_file = Path(project_name) / "temp_data" / f"advantage_stats_epoch_{current_epoch}.txt"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        f.write(f"Epoch: {current_epoch}\n")
        strategy_name = "Batch-wise" if adv_norm_mode == "batch" else "Prompt-wise"
        f.write(f"Strategy: {strategy_name} Normalized (No Value Model)\n")
        f.write(f"Adv Norm Mode: {adv_norm_mode}\n")
        f.write(f"Num responses per task: {num_responses_per_task}\n")
        f.write(f"Num prompts: {num_prompts}\n")
        f.write(f"Num sequences: {B}\n")
        f.write(f"Num nonzero advantages: {num_nonzero_adv}\n")
        f.write(f"Mean advantage: {mean_adv:.4f}\n")
        f.write(f"Std advantage: {std_adv:.4f}\n")
        f.write(f"Collapse Ratios: 0.01={collapse_ratios[0.01]:.4f}, 0.05={collapse_ratios[0.05]:.4f}, 0.1={collapse_ratios[0.1]:.4f}, 0.5={collapse_ratios[0.5]:.4f}\n")


def main():
    """
    Main function: load data from JSON, process it, compute advantages, save PT file.
    """
    config = get_config()
    project_name = config.experiment.project
    current_epoch = config.experiment.current_epoch

    logger.info("="*80)
    logger.info(f"[No Value Model] Starting data processing for epoch {current_epoch}")
    logger.info("="*80)

    # 1. Load tokenizer
    if current_epoch <= 1:
        pretrained_model = config.model.pretrained_model
    else:
        # Use relative path (consistent with train_policy.py)
        pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_name

    logger.info(f"[No Value Model] Loading tokenizer from {pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True, local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    # Check for critical issue: pad_token_id == eos_token_id
    # Warning only - do NOT auto-fix (auto-fix causes inconsistency with data loading)
    if pad_id == eos_id:
        logger.warning("=" * 80)
        logger.warning("WARNING: pad_token_id == eos_token_id detected!")
        logger.warning(f"pad_token_id = {pad_id}, eos_token_id = {eos_id}")
        logger.warning("Using original pad_id with EOS handling logic.")
        logger.warning("=" * 80)

    # 2. Load rollout data from JSON
    json_path = Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}.json"
    if not json_path.exists():
        logger.error(f"JSON data file not found: {json_path}")
        raise FileNotFoundError(f"JSON data file not found: {json_path}")

    logger.info(f"[No Value Model] Loading JSON data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 3. Extract data fields
    prompt_list = [x["prompt"] for x in json_data]
    response_list = [x["response"] for x in json_data]
    reward_list = [x["reward"] for x in json_data]
    step_map_list = [x.get("step_map", []) for x in json_data]
    correctness_list = [x.get("correctness", x["reward"] >= 0) for x in json_data]
    sample_idx_list = [x.get("sample_idx", -1) for x in json_data]
    resp_idx_list = [x.get("resp_idx", -1) for x in json_data]
    truncated_list = [x.get("truncated", False) for x in json_data]

    logger.info(f"[No Value Model] Loaded {len(json_data)} samples from JSON")

    # 4. Call UniversalPrompting to process data
    logger.info(f"[No Value Model] Processing prompts with UniversalPrompting")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100
    )

    input_ids_lm, _, start_pos, drop_num, keep_indices = uni_prompting(
        (prompt_list, response_list)
    )
    start_pos = int(start_pos)

    logger.info(f"[No Value Model] After uni_prompting: {input_ids_lm.shape[0]} samples, dropped {drop_num}")

    # 5. Filter other lists based on keep_indices
    if drop_num > 0:
        prompt_list = [prompt_list[i] for i in keep_indices]
        response_list = [response_list[i] for i in keep_indices]
        reward_list = [reward_list[i] for i in keep_indices]
        correctness_list = [correctness_list[i] for i in keep_indices]
        sample_idx_list = [sample_idx_list[i] for i in keep_indices]
        resp_idx_list = [resp_idx_list[i] for i in keep_indices]
        step_map_list = [step_map_list[i] for i in keep_indices]
        json_data = [json_data[i] for i in keep_indices]

    # 6. Process step_map, ensure consistent lengths
    _, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.post_num

    for i, x in enumerate(json_data):
        if "step_map" not in x or not x["step_map"]:
            step_map_list[i] = [j for j in range(L1)]
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list[i] = step_map_i

    # 7. Build training data tensors
    B = input_ids_lm.shape[0]
    device = input_ids_lm.device

    # step_map_all and resp_input_ids_all
    step_map_all = torch.tensor(step_map_list, dtype=torch.long, device=device)  # (B, L1)
    resp_input_ids_all = input_ids_lm[:, L0:].clone()  # (B, L1)

    # per_seq_reward
    per_seq_reward = torch.tensor(reward_list, dtype=torch.float32)  # (B,)

    # correctness_all
    correctness_all = torch.tensor(correctness_list, dtype=torch.bool)  # (B,)

    # sample_idx_all and resp_idx_all
    sample_idx_all = torch.tensor(sample_idx_list, dtype=torch.long)  # (B,)
    resp_idx_all = torch.tensor(resp_idx_list, dtype=torch.long)  # (B,)

    # extended_input_ids (use input_ids_lm directly)
    extended_input_ids = input_ids_lm  # (B, L)

    # p_mask: marks positions where logits should be extracted (response part only)
    # Prompt portion: False (Block Diffusion does not output logits for prompt)
    # Response portion: True for non-padding positions and positions after EOS
    p_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Prompt portion: excluded (Block Diffusion model constraint)
    p_mask[:, :L0] = False

    # Response portion: True for non-padding positions
    resp_ids = input_ids_lm[:, L0:L0+L1]  # (B, L1)
    is_pad = resp_ids.eq(pad_id)

    # Handle EOS tokens: exclude tokens after the first EOS
    if eos_id is not None:
        is_eos = resp_ids.eq(eos_id)
        # Create cumulative mask: positions after EOS are True
        eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)
        # Find the first EOS position in each sequence
        eos_pos = is_eos.float().argmax(dim=1)
        # Mark positions after EOS (excluding EOS itself, which should participate in training)
        for b in range(B):
            if is_eos[b].any():
                first_eos_pos = eos_pos[b].item()
                # Tokens after EOS should not participate in training
                eos_after_mask[b, first_eos_pos+1:] = True
    else:
        eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)

    # Combine masks: exclude padding and tokens after EOS
    if post_num is None or post_num == 0:
        resp_trainable = ~is_pad & ~eos_after_mask
    else:
        cum_pad = torch.cumsum(is_pad.int(), dim=1)
        resp_trainable = (~is_pad & ~eos_after_mask) | (is_pad & (cum_pad <= post_num) & ~eos_after_mask)

    p_mask[:, L0:L0+L1] = resp_trainable

    # tok_idx_ext: token index at each position
    tok_idx_ext = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)

    # labels: compute loss only for response portion (set padding positions to -100)
    labels = input_ids_lm.clone()
    labels[:, :L0] = -100  # Do not compute loss for prompt portion

    # Assign directly to response portion to avoid dimension issues
    # Create mask for response portion, then apply
    resp_mask = ~is_pad  # (B, L1) True for non-padding positions
    labels[:, L0:L0+L1] = torch.where(resp_mask, input_ids_lm[:, L0:L0+L1], torch.tensor(-100, device=device))

    # seq_ids: one row per sequence
    seq_ids = torch.arange(B, dtype=torch.long, device=device)  # (B,)

    # Initialize Return and adv
    Return_mat = torch.zeros(B, L, dtype=torch.float32, device=device)
    adv_mat = torch.zeros(B, L, dtype=torch.float32, device=device)

    # 8. Compute advantages
    adv_norm_mode = OmegaConf.select(config, "reward.adv_norm_mode", default="batch")

    compute_advantages_no_value(
        Return_mat, adv_mat, per_seq_reward, seq_ids, p_mask, L0,
        project_name, current_epoch, input_ids_lm, eos_id, L1,
        sample_idx_all=sample_idx_all,  # For correct grouping
        resp_idx_all=resp_idx_all,      # For correct grouping
        adv_norm_mode=adv_norm_mode     # Advantage normalization mode
    )

    # 9. Save as PT file for policy training
    output_path = Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}.pt"
    output_data = {
        "extended_input_ids": extended_input_ids.cpu(),
        "p_mask": p_mask.cpu(),
        "tok_idx_ext": tok_idx_ext.cpu(),
        "labels": labels.cpu(),
        "adv": adv_mat.cpu(),  # Updated advantage
        "Return": Return_mat.cpu(),  # Updated return
        "correctness_all": correctness_all.cpu(),
        "L0": L0,
        "L1": L1,
        "pad_id": pad_id,
        "post_num": post_num,
        "step_map_all": step_map_all.cpu(),
        "resp_input_ids_all": resp_input_ids_all.cpu(),
        "per_seq_reward": per_seq_reward.cpu(),
        "seq_ids": seq_ids.cpu(),
        "step_rewards_all": None,
        "sample_idx_all": sample_idx_all.cpu(),
        "resp_idx_all": resp_idx_all.cpu(),
        "meta": {
            "epoch": current_epoch,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_pos": L0,
            "drop_num": drop_num,
        }
    }

    torch.save(output_data, output_path)
    logger.info(f"[No Value Model] Saved data to {output_path}")

    logger.info("="*80)
    logger.info(f"[No Value Model] Data processing completed for epoch {current_epoch}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
