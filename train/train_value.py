
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ===== Suppress All Warnings and Logs =====
# Environment variables (must be set before importing libraries)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Set terminal width to fix tqdm progress bar display
os.environ.setdefault("COLUMNS", "120")

import logging
# Suppress torch._inductor autotune logs
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad.*")
warnings.filterwarnings("ignore", message=".*is part of.*")
warnings.filterwarnings("ignore", message=".*docstring.*")
# ==========================
import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from transformers import AutoModelForCausalLM, AutoTokenizer
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from models.value_metrics import compute_value_metrics, format_value_metrics, collect_value_metrics_from_dataset
from termcolor import cprint

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter


# ===== Helper Functions for Trainable Mask Computation =====
def compute_trainable_mask_for_row(dataset, row_idx, L0, L1):
    """Compute trainable mask for a single row, including all non-padding response tokens.

    Args:
        dataset: TrainDataset instance
        row_idx: int, row index in dataset
        L0: int, prompt length
        L1: int, response length

    Returns:
        full_mask: (L0 + L1,) bool tensor, True means the position is trainable
    """
    # Get the original sequence for this row
    s = int(dataset.seq_ids[row_idx].item())
    resp_ids = dataset.resp_input_ids_all[s]  # (L1,)

    is_pad = resp_ids.eq(dataset.pad_id)

    # Add EOS handling: exclude tokens after the first EOS
    eos_id = getattr(dataset, 'eos_id', None)
    eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)
    if eos_id is not None:
        is_eos = resp_ids.eq(eos_id)
        if is_eos.any():
            first_eos_pos = is_eos.float().argmax(dim=0).item()
            eos_after_mask[first_eos_pos+1:] = True

    if dataset.post_num is None:
        trainable = ~is_pad & ~eos_after_mask
    else:
        cum_pad = torch.cumsum(is_pad.int(), dim=0)
        trainable = (~is_pad & ~eos_after_mask) | (is_pad & (cum_pad <= dataset.post_num) & ~eos_after_mask)

    # Expand to full sequence (L0 + L1)
    full_mask = torch.zeros(L0 + L1, dtype=torch.bool)
    full_mask[:L0] = True  # Prompt portion
    full_mask[L0:L0+L1] = trainable  # Response portion
    return full_mask


def compute_trainable_mask_for_batch(labels, pad_id, L0, post_num):
    """Compute trainable mask for a batch, including all non-padding response tokens.

    Args:
        labels: (B, L) tensor, target labels
        pad_id: int, padding token id
        L0: int, prompt length
        post_num: int or None, number of padding tokens to include

    Returns:
        trainable_mask: (B, L) bool tensor, True means the position is trainable
    """
    B, L = labels.shape
    device = labels.device

    # Prompt portion - all True (value prediction not needed, but kept consistent)
    trainable = torch.arange(L, device=device).unsqueeze(0).expand(B, -1) < L0

    # Response portion - exclude padding (unless within post_num range)
    if L > L0:
        is_pad = (labels[:, L0:] == pad_id)
        if post_num is None:
            trainable_response = ~is_pad
        else:
            cum_pad = torch.cumsum(is_pad.int(), dim=1)
            trainable_response = (~is_pad) | (is_pad & (cum_pad <= post_num))

        # Assign directly to response portion, maintaining (B, L) dimensions
        trainable[:, L0:] = trainable_response

    return trainable
# ===== End Helper Functions =====

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def log_gpu_memory(accelerator, message):
    """Record GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"[GPU {i}] {message}: "
                      f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                      f"Max={max_allocated:.2f}GB, Total={total:.2f}GB")


# ===== Flex Attention Support =====
# Import flex attention - will error if not available
from torch.nn.attention.flex_attention import create_block_mask, BlockMask


def create_flex_block_mask(attention_mask_bool):
    """Convert bool tensor to BlockMask for flex_attention.

    Args:
        attention_mask_bool: bool tensor (B, N, N) where True = attend

    Returns:
        BlockMask object
    """
    B, N = attention_mask_bool.shape[:2]
    return create_block_mask(
        lambda b, h, q_idx, kv_idx: attention_mask_bool[b, q_idx, kv_idx],
        B=B, H=None, Q_LEN=N, KV_LEN=N,
    )
# ===== End Flex Attention Support =====






class TrainDataset(Dataset):
    def __init__(
        self,
        extended_input_ids, p_mask, tok_idx_ext, labels, reward,
        *,
        seq_ids,                  # (N_rows,)  The original sequence id (i.e., batch index b) for this row
        L0, L1,                   # Prompt length, response length
        step_map_all,             # (B, L1)    Step map for each original sequence (already shrunk)
        resp_input_ids_all,       # (B, L1)    Response token ids for each original sequence (used for pad/post_num)
        per_seq_reward,           # (B,)       Scalar reward for each original sequence
        pad_id, post_num,         # Pad token id and post_num
        eos_id=None,              # EOS token id for EOS handling logic
        step_rewards_all=None,    # (B,) list of dicts, per-step rewards
        correctness_all=None,     # (B,) bool tensor, correctness for each original sequence
        sample_idx_all=None,      # (N_rows,) Original sample indices for KL penalty mapping
        resp_idx_all=None         # (N_rows,) Response indices for KL penalty mapping
    ):
        self.extended_input_ids = extended_input_ids  # (N_rows, L_ext)
        self.p_mask  = p_mask                         # (N_rows, L=L0+L1), only valid for the first L positions
        self.tok_idx_ext = tok_idx_ext               # (N_rows, L)
        self.labels  = labels                        # (N_rows, L)
        self.Return  = reward                        # (N_rows, L) Placeholder, will be overwritten with "return" later
        # --- Extra fields for aggregation ---
        self.seq_ids = torch.as_tensor(seq_ids, dtype=torch.long)        # (N_rows,)
        self.L0 = int(L0); self.L1 = int(L1)
        self.step_map_all = step_map_all.clone().cpu()                   # (B, L1)
        self.resp_input_ids_all = resp_input_ids_all.clone().cpu()       # (B, L1)
        self.per_seq_reward = torch.as_tensor(per_seq_reward, dtype=torch.float32)  # (B,)
        self.pad_id = int(pad_id); self.post_num = int(post_num) if post_num is not None else None
        self.eos_id = eos_id  # Store eos_id for EOS handling logic

        # Step-level reward support
        self.step_rewards_all = step_rewards_all  # list of dicts: [{step_id: reward}, ...]

        # Correctness for NLL loss (VAPO)
        if correctness_all is not None:
            self.correctness_all = torch.as_tensor(correctness_all, dtype=torch.bool)  # (B,)
        else:
            self.correctness_all = None

        # Sample and response indices for KL penalty mapping
        if sample_idx_all is not None:
            self.sample_idx_all = torch.as_tensor(sample_idx_all, dtype=torch.long)  # (N_rows,)
        else:
            self.sample_idx_all = None
        if resp_idx_all is not None:
            self.resp_idx_all = torch.as_tensor(resp_idx_all, dtype=torch.long)  # (N_rows,)
        else:
            self.resp_idx_all = None

        # Old values (predicted by the model during inference)
        self.old_values = torch.full((len(extended_input_ids), p_mask.shape[1]), 0.0)
        # Advantage placeholder (to be filled later)
        self.adv = torch.full((len(extended_input_ids), p_mask.shape[1]), 0.0)
        # KL penalty for KL-in-reward (VAPO): token-level KL = log_prob_policy - log_prob_ref
        self.kl_penalty = torch.zeros((len(extended_input_ids), p_mask.shape[1]), dtype=torch.float32)

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.Return[idx],   
        )





def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_value_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    # Shared backbone mode: ValueModel wraps SDARForCausalLM with an additional value_head
    from models import SDARForCausalLM
    from train.init_value_model import _get_value_model
    value_model_class = _get_value_model(SDARForCausalLM, "value_head")
    value_model = value_model_class.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # KL-in-reward: KL penalty is now pre-computed during rollout phase
    # This is more efficient: no need to load policy/ref models here
    kl_in_reward_enabled = OmegaConf.select(config, "kl_in_reward.enabled", default=False)
    current_epoch = config.experiment.current_epoch

    # Determine if we're in value pretraining mode
    value_pretrain_steps = OmegaConf.select(config, "training.value_pretrain_steps", default=0)
    value_pretrain_enabled = OmegaConf.select(config, "training.value_pretrain_enabled", default=False)
    # Pretraining if steps-based condition is met
    # Note: if value_pretrain_enabled=True, convergence will be checked in training loop
    is_value_pretraining = value_pretrain_steps > 0 and current_epoch <= value_pretrain_steps

    # Load pre-computed KL penalty from file (if available)
    kl_penalty_data = None
    if kl_in_reward_enabled and current_epoch > 1:
        kl_file = os.path.join(project_name, "temp_data", "kl_penalty.pt")
        if os.path.exists(kl_file):
            kl_data = torch.load(kl_file)
            # New format: dict with "kl_results" key
            if isinstance(kl_data, dict) and "kl_results" in kl_data:
                kl_penalty_data = kl_data["kl_results"]
                logger.info(f"Loaded pre-computed KL penalty from {kl_file} (nested format)")
            # Old format: directly the nested dict
            elif isinstance(kl_data, dict):
                kl_penalty_data = kl_data
                logger.info(f"Loaded pre-computed KL penalty from {kl_file} (old format)")
            else:
                logger.warning(f"Unexpected KL penalty format, KL will be 0")
        else:
            logger.warning(f"KL penalty file not found: {kl_file}, KL will be 0")
    elif kl_in_reward_enabled and current_epoch <= 1:
        logger.info("Epoch <= 1: policy == ref, KL = 0")

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    
    
    
    
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)
    

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(value_model, "config"):
        value_model.config.fuse_cross_entropy = False   
    

    if config.training.gradient_checkpointing_enable:
        value_model.gradient_checkpointing_enable()
        if hasattr(value_model, "config"):
            value_model.config.use_cache = False
    else:
        value_model = value_model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    # Check for critical issue: pad_token_id == eos_token_id in value training
    # Warning only - do NOT auto-fix (auto-fix causes inconsistency with data loading)
    if pad_id == eos_id:
        logger.warning("=" * 80)
        logger.warning("WARNING: pad_token_id == eos_token_id detected!")
        logger.warning(f"pad_token_id = {pad_id}, eos_token_id = {eos_id}")
        logger.warning("Using original pad_id with EOS handling logic.")
        logger.warning("=" * 80)

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in value_model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in value_model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.value_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")




    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    







    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, Return = zip(*batch)     #Tensor(L)
        return {
            "ids":        torch.tensor(idx),
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels),
            "Return":     torch.stack(Return),
        }
    


    
    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]


    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    step_rewards_list = []  # List of dicts for step-level reward
    correctness_list = []   # List of bool for NLL loss (VAPO)
    sample_idx_list = []    # List of sample indices for KL penalty mapping
    resp_idx_list = []      # List of response indices for KL penalty mapping
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
        # Load step_rewards if available
        if "step_rewards" in x:
            # Convert string keys back to int
            step_rewards_dict = {int(k): v for k, v in x["step_rewards"].items()}
            step_rewards_list.append(step_rewards_dict)
        else:
            step_rewards_list.append({})
        # Load correctness if available (for NLL loss)
        if "correctness" in x:
            correctness_list.append(x["correctness"])
        else:
            # Fall back to reward >= 0 as correctness indicator
            correctness_list.append(x["reward"] >= 0)
        # Load sample_idx and resp_idx for KL penalty mapping
        sample_idx_list.append(x.get("sample_idx", -1))
        resp_idx_list.append(x.get("resp_idx", -1))
    input_ids_lm, _, start_pos, drop_num, keep_indices = uni_prompting((prompt_list, response_list))
    start_pos = int(start_pos)

    # Sync filter all lists based on keep_indices to ensure index alignment
    # This is critical for KL penalty mapping to work correctly
    if drop_num > 0:
        prompt_list = [prompt_list[i] for i in keep_indices]
        response_list = [response_list[i] for i in keep_indices]
        sample_idx_list = [sample_idx_list[i] for i in keep_indices]
        resp_idx_list = [resp_idx_list[i] for i in keep_indices]
        step_rewards_list = [step_rewards_list[i] for i in keep_indices]
        correctness_list = [correctness_list[i] for i in keep_indices]
        reward_list = [reward_list[i] for i in keep_indices]
        # Also need to filter dataset_load for step_map_list construction
        dataset_load = [dataset_load[i] for i in keep_indices]
        logger.info(f"[uni_prompting] Filtered {drop_num} samples, {len(dataset_load)} remaining")

    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num


    for x in dataset_load:
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)
    
    def create_block_mask_for_batch(
        B: int,
        L: int,
        L0: int,
        block_size: int,
        device: torch.device,
        pad_id: int,
        input_ids: torch.Tensor,
    ) -> "BlockMask":
        """Dynamically create block_mask for each batch based on actual input size.

        This function combines the logic of make_basic_block_attention and process_pad,
        creating a BlockMask object that matches the actual batch dimensions.

        Args:
            B: batch size
            L: total sequence length (L0 + L1)
            L0: prompt length (start_pos)
            block_size: block size for block attention
            device: torch device
            pad_id: padding token id
            input_ids: input token ids (B, L) for padding detection

        Returns:
            BlockMask object for flex_attention
        """
        L1 = L - L0  # Response length
        N = L  # Total length

        # Initialize with False, then set to True for valid attention
        bias = torch.full((B, 1, N, N), False, dtype=torch.bool, device=device)

        # For value model training, we only have one response part [L0, N)
        # Create block attention pattern for the response tokens
        if L1 > 0:
            # Response tokens
            rows_token = torch.arange(L0, N, device=device)  # [L0, N) response tokens

            # Update block by block
            for bi in range((L1 + block_size - 1) // block_size):
                i_start = bi * block_size
                i_end = min((bi + 1) * block_size, L1)

                # Current block's response token rows
                left_end = L0 + i_start
                right_end = L0 + i_end

                if left_end < N:
                    block_rows = rows_token[i_start:i_end]
                    # Each token can attend to all tokens up to current block end
                    bias[:, :, block_rows.unsqueeze(-1), 0:right_end] = True

        # Prompt part: full attention (causal within prompt)
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                row_end = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end, device=device)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = True

        # Squeeze head dimension: (B, 1, N, N) -> (B, N, N)
        # This allows 2D key_mask to index the tensor correctly
        bias = bias.squeeze(1)  # (B, N, N)

        # Process padding: mask out padding positions in prompt
        # Now bias is 3D (B, N, N), so 2D key_mask can index correctly
        cols = torch.arange(N, device=device)  # (N,)
        key_mask = (cols < L0).unsqueeze(0) & (input_ids == pad_id)  # (B, N)
        bias[key_mask, :] = False  # ✅ 3D tensor with 2D mask

        # Ensure each token can attend to itself
        # Fix rows that have no attention by enabling self-attention
        row_has_no_attention = (bias.sum(dim=-1) == 0)  # (B, N)
        col_before_start = torch.arange(N, device=device) < L0  # (N,)
        bad = row_has_no_attention & col_before_start.unsqueeze(0)  # (B, N)

        # Set diagonal self-attention for rows with no attention
        b_idx, r_idx = bad.nonzero(as_tuple=True)  # 2D tensor → 2 return values
        if len(b_idx) > 0:
            bias[b_idx, r_idx, r_idx] = True  # Diagonal indexing: batch, row, col

        # Convert to BlockMask for flex_attention
        # bias is already 3D (B, N, N) after squeezing earlier
        return create_block_mask(
            lambda b, h, q_idx, kv_idx: bias[b, q_idx, kv_idx],
            B=B,
            H=None,
            Q_LEN=N,
            KV_LEN=N,
        )




    def collect_training_data(input_ids, step_map_list, reward,
                             sample_idx_list=None, resp_idx_list=None):
        """Collect training data, creating 1 row per sequence with all trainable positions.

        Aligned with standard LLM PPO: train all non-padding response tokens,
        not just the first token per step (block-wise selection).

        Args:
            input_ids: (B, L) input token ids
            step_map_list: list of lists, step map for each sequence
            reward: (B,) scalar reward for each sequence
            sample_idx_list: list of sample indices for KL penalty mapping
            resp_idx_list: list of response indices for KL penalty mapping

        Returns:
            extended_input_ids: (B, L) input token ids (truncated, no block diffusion extension)
            p_mask: (B, L) bool mask, True means the position is trainable
            tok_idx_ext: (B, L) token indices
            labels: (B, L) target labels
            reward_mat: (B, L) reward matrix (all zeros, filled later)
            seq_ids_rows: (B,) sequence ids
            sel_step_tail_rows: (B, L1) step ids (for compatibility, all -1)
            step_map: (B, L1) step map for each original sequence
            resp_input_ids_all: (B, L1) response token ids
            sample_idx_rows: (B,) sample indices for KL penalty mapping
            resp_idx_rows: (B,) response indices for KL penalty mapping
        """
        B, L = input_ids.shape
        L0 = start_pos
        L1 = L - L0
        block_size = config.training.block_size

        # shrink (in-place modification of step_map_list)
        # Keep this for now as it affects step_map used in compute_returns_and_advantages
        for b in range(B):
            step_map_i = step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                s = j * block_size
                e = min(L1, (j + 1) * block_size)
                step_map_list[b][s:e] = collapse_k_unique(step_map_i[s:e], config.training.shrink)

        step_map = torch.as_tensor(step_map_list, dtype=torch.long)  # (B, L1)
        assert step_map.shape[1] == L1

        # Initialize output tensors (1 row per sequence)
        extended_input_ids_list = []
        pmask_list = []
        seq_ids_rows = []
        sel_step_tail_rows = []
        reward_list_rows = []
        sample_idx_rows = [] if sample_idx_list is not None else None
        resp_idx_rows = [] if resp_idx_list is not None else None

        for b in range(B):
            # Create 1 row per sequence with all trainable positions marked
            input_ids_b = input_ids[b]  # (L,)

            # Compute trainable mask: all non-padding response tokens
            # Response segment is [L0, L0+L1)
            resp_ids = input_ids_b[L0:L0+L1]  # (L1,)
            is_pad = resp_ids.eq(pad_id)

            # Handle EOS tokens: exclude tokens after the first EOS
            if eos_id is not None:
                is_eos = resp_ids.eq(eos_id)
                eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)
                if is_eos.any():
                    first_eos_pos = is_eos.float().argmax(dim=0).item()
                    eos_after_mask[first_eos_pos+1:] = True
            else:
                eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)

            if post_num is None:
                resp_trainable = ~is_pad & ~eos_after_mask
            else:
                cum_pad = torch.cumsum(is_pad.int(), dim=0)
                resp_trainable = (~is_pad & ~eos_after_mask) | (is_pad & (cum_pad <= post_num) & ~eos_after_mask)

            # Build p_mask: (L,) bool tensor
            # Prompt part: all True (we don't compute value for prompt, but keep for consistency)
            # Response part: trainable mask
            p_mask_b = torch.zeros(L, dtype=torch.bool, device=input_ids.device)
            p_mask_b[:L0] = True  # prompt part
            p_mask_b[L0:L0+L1] = resp_trainable  # response part

            # Store the data
            extended_input_ids_list.append(input_ids_b)
            pmask_list.append(p_mask_b)
            seq_ids_rows.append(b)
            # sel_step_tail_rows: all -1 for compatibility (not used in new flow)
            sel_step_tail_rows.append(torch.full((L1,), -1, dtype=torch.long))
            reward_list_rows.append(reward[b])

            # Collect indices for KL penalty mapping
            if sample_idx_rows is not None:
                sample_idx_rows.append(sample_idx_list[b])
            if resp_idx_rows is not None:
                resp_idx_rows.append(resp_idx_list[b])

        # Stack all tensors
        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)  # (B, L)
        p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)  # (B, L)
        seq_ids_rows = torch.as_tensor(seq_ids_rows, dtype=torch.long)  # (B,)
        sel_step_tail_rows = torch.stack(sel_step_tail_rows, dim=0)  # (B, L1)

        # Note: post_num is already handled above, no need to apply again

        # Truncate to L positions (no block diffusion extension needed for value model)
        extended_input_ids = extended_input_ids[:, :L].contiguous()  # (B, L)
        labels = extended_input_ids.clone()

        # Compute tok_idx_ext
        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids.ne(pad_id)
        tok_idx = valid.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_ext = tok_idx  # (B, L)

        # Filter out rows where nothing was selected (shouldn't happen if data is valid)
        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx_keep = keep.nonzero(as_tuple=True)[0]

        extended_input_ids = extended_input_ids[idx_keep]
        p_mask = p_mask[idx_keep]
        tok_idx_ext = tok_idx_ext[idx_keep]
        labels = labels[idx_keep]
        seq_ids_rows = seq_ids_rows[idx_keep]
        sel_step_tail_rows = sel_step_tail_rows[idx_keep]
        reward_rows = [reward_list_rows[i] for i in idx_keep.tolist()]

        # Filter index rows for KL penalty mapping
        if sample_idx_rows is not None:
            sample_idx_rows = [sample_idx_rows[i] for i in idx_keep.tolist()]
        if resp_idx_rows is not None:
            resp_idx_rows = [resp_idx_rows[i] for i in idx_keep.tolist()]

        # Initialize reward_mat with all zeros
        reward_vec = torch.as_tensor(reward_rows, dtype=torch.float32, device=p_mask.device)
        reward_mat = torch.zeros_like(p_mask, dtype=torch.float32)

        # Extra return: for later aggregation
        resp_input_ids_all = input_ids[:, L0:L0+L1].clone().cpu()  # (B, L1)

        return (
            extended_input_ids, p_mask, tok_idx_ext, labels, reward_mat,
            seq_ids_rows, sel_step_tail_rows, step_map, resp_input_ids_all,
            sample_idx_rows, resp_idx_rows
        )

        


    (extended_input_ids, p_mask, tok_idx_ext, labels, rewards,        # rewards all 0, as place-holder
        seq_ids_rows, sel_step_tail_rows, step_map_all, resp_input_ids_all,
        sample_idx_rows, resp_idx_rows) = collect_training_data(
            input_ids_lm, step_map_list, reward_list,
            sample_idx_list=sample_idx_list,
            resp_idx_list=resp_idx_list
        )



    dataset_lm = TrainDataset(
        extended_input_ids, p_mask, tok_idx_ext, labels, rewards,
        seq_ids=seq_ids_rows,
        L0=start_pos, L1=(labels.shape[1]-start_pos),
        step_map_all=step_map_all,
        resp_input_ids_all=resp_input_ids_all,
        per_seq_reward=torch.as_tensor(reward_list, dtype=torch.float32),
        pad_id=pad_id, post_num=post_num,
        eos_id=eos_id,
        step_rewards_all=step_rewards_list,
        correctness_all=correctness_list,
        sample_idx_all=sample_idx_rows,  # Use indices from collect_training_data (expanded rows)
        resp_idx_all=resp_idx_rows        # Use indices from collect_training_data (expanded rows)
    )

    # Validate index length alignment
    if dataset_lm.sample_idx_all is not None and dataset_lm.resp_idx_all is not None:
        assert len(dataset_lm.sample_idx_all) == len(dataset_lm), \
            f"sample_idx_all length {len(dataset_lm.sample_idx_all)} != dataset length {len(dataset_lm)}"
        assert len(dataset_lm.resp_idx_all) == len(dataset_lm), \
            f"resp_idx_all length {len(dataset_lm.resp_idx_all)} != dataset length {len(dataset_lm)}"
        assert dataset_lm.sample_idx_all.min() >= 0, \
            f"sample_idx_all contains negative values"
        logger.info(f"[Index Validation] sample_idx_all/resp_idx_all length matched: {len(dataset_lm)} rows")

    # Load pre-computed KL penalty from rollout phase
    # The KL is indexed by (sample_idx, response_idx) and needs to be mapped to dataset rows
    if kl_penalty_data is not None and kl_in_reward_enabled:
        logger.info("Filling KL penalty from pre-computed data...")
        L0 = start_pos
        L1 = labels.shape[1] - start_pos

        # Map KL data to dataset rows using sample_idx_all and resp_idx_all
        # kl_penalty_data structure: {sample_idx: {response_idx: kl_tensor}}
        # dataset_lm.sample_idx_all and dataset_lm.resp_idx_all contain the original indices
        if dataset_lm.sample_idx_all is not None and dataset_lm.resp_idx_all is not None:
            filled_count = 0
            missing_count = 0
            for row_idx in range(len(dataset_lm)):
                sample_idx = int(dataset_lm.sample_idx_all[row_idx].item())
                response_idx = int(dataset_lm.resp_idx_all[row_idx].item())

                if sample_idx >= 0 and sample_idx in kl_penalty_data and response_idx in kl_penalty_data[sample_idx]:
                    kl_tensor = kl_penalty_data[sample_idx][response_idx]
                    # Get p_mask for this row to determine where to fill KL
                    pm = dataset_lm.p_mask[row_idx]  # (L0+L1,)
                    tail_mask = pm[L0:L0+L1]  # Only response part

                    # KL tensor from rollout is for the full sequence, we need to extract response part
                    # The KL tensor should align with the tokenized response
                    if len(kl_tensor) >= L1:
                        # Fill KL values at p_mask positions
                        kl_values = kl_tensor[-L1:] if len(kl_tensor) > L1 else kl_tensor
                        dataset_lm.kl_penalty[row_idx, L0:L0+L1] = kl_values[:L1]
                    filled_count += 1
                else:
                    missing_count += 1
            logger.info(f"[KL Penalty] Filled: {filled_count}/{len(dataset_lm)} rows ({filled_count/len(dataset_lm)*100:.1f}%)")
            if missing_count > 0:
                logger.warning(f"[KL Penalty] Missing: {missing_count}/{len(dataset_lm)} rows (KL penalty not found for these indices)")
        else:
            # Fallback: use seq_ids (may not work correctly with data_filter)
            logger.warning("sample_idx_all or resp_idx_all is None, using seq_ids as fallback (may be incorrect with data_filter)")
            filled_count = 0
            for row_idx in range(len(dataset_lm)):
                seq_id = int(dataset_lm.seq_ids[row_idx].item())
                k_sample = config.rollout.num_response_per_task
                sample_idx = seq_id // k_sample
                response_idx = seq_id % k_sample

                if sample_idx in kl_penalty_data and response_idx in kl_penalty_data[sample_idx]:
                    kl_tensor = kl_penalty_data[sample_idx][response_idx]
                    pm = dataset_lm.p_mask[row_idx]
                    if len(kl_tensor) >= L1:
                        kl_values = kl_tensor[-L1:] if len(kl_tensor) > L1 else kl_tensor
                        dataset_lm.kl_penalty[row_idx, L0:L0+L1] = kl_values[:L1]
                    filled_count += 1
            logger.info(f"[KL Penalty] Filled (fallback): {filled_count}/{len(dataset_lm)} rows ({filled_count/len(dataset_lm)*100:.1f}%)")

    # Use batch_size_value if specified, otherwise fall back to batch_size_lm for compatibility
    batch_size_value = OmegaConf.select(config, "training.batch_size_value",
                                         default=OmegaConf.select(config, "training.batch_size_lm", default=8))

    # Use batch_size_value_inference for compute_old_value_parallel (can be larger than training batch size)
    batch_size_value_inference = OmegaConf.select(config, "training.batch_size_value_inference",
                                                   default=batch_size_value)  # fallback to batch_size_value

    total_batch_size_lm = batch_size_value * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    config.lr_scheduler.params.learning_rate = config.optimizer.params.value_learning_rate
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    # Single DataLoader for both training and inference
    # Use larger batch_size for inference efficiency
    # Training will manually split batches to smaller size if needed
    dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=batch_size_value_inference,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )





    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    value_model, optimizer, lr_scheduler, dataloader_lm = accelerator.prepare(
        value_model, optimizer, lr_scheduler, dataloader_lm
    )
    # Ensure model is in training mode (needs reset after DeepSpeed wrapping, otherwise self.training will be False)
    # Both DeepSpeed and Accelerate use the 'module' attribute
    # The train() method of DeepSpeed and Accelerate automatically propagates to inner modules
    getattr(value_model, 'module', value_model).train()

    import torch.nn.functional as F





    @torch.no_grad()
    def compute_old_value_parallel(
            accelerator,
            dataset,
            dataloader,
            start_pos, pad_id):
        """
        Optimized: accumulate locally during loop, gather only once at the end.
        This reduces sync overhead from 2N to 2 (where N = number of batches).
        Note: Uses train() mode instead of eval() to support flex attention.
        """
        # DeepSpeed direct assignment requires explicitly setting training state of inner modules
        getattr(value_model, 'module', value_model).train()
        dl = dataloader

        # Local accumulation lists (no sync during loop)
        local_ids_list = []
        local_values_list = []

        for batch in dl:
            ids        = batch["ids"]  # (b,)
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)

            B, L = p_mask.shape
            L0 = start_pos
            L1 = L - L0
            device = extended_input_ids.device

            # Synchronize position_ids construction (consistent with policy training)
            # valid_indices: 1=prompt, 2=response, 0=padding
            valid_indices = torch.zeros(B, L, dtype=torch.long, device=device)
            valid_indices[:, :L0] = 1   # Set prompt portion to 1
            valid_indices[:, L0:L0+L1] = 2  # Set response portion to 2
            # Padding portion stays 0
            position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
            position_ids = torch.where(valid_indices == 0, torch.zeros_like(position_ids), position_ids)

            # ===== Attention Mask Creation =====
            # Dynamically create block_mask for this batch based on actual input size
            # Note: Flex Attention is compatible with train() + no_grad mode
            attention_mask = create_block_mask_for_batch(
                B=B, L=L, L0=L0,
                block_size=config.training.block_size,
                device=device,
                pad_id=pad_id,
                input_ids=extended_input_ids,
            )
            # ===== End Attention Mask =====

            values = value_model(
                input_ids=extended_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids  # Use modified position_ids (consistent with policy training)
            )
            values = values[:, :L0+L1]  # (B, L0+L1) - keep original response positions
            values = torch.where(p_mask, values, torch.zeros_like(values))

            # Accumulate locally (CPU) - no sync!
            local_ids_list.append(ids.cpu())
            local_values_list.append(values.float().cpu())

        # After loop: one-time gather (if multi-process)
        if accelerator.num_processes > 1:
            # Concatenate local results
            local_ids = torch.cat(local_ids_list, dim=0)        # (local_N,)
            local_values = torch.cat(local_values_list, dim=0)  # (local_N, L)
            
            # Move to device for gather
            local_ids_dev = local_ids.to(accelerator.device)
            local_values_dev = local_values.to(accelerator.device)
            
            # Pad across processes (one sync)
            ids_pad = accelerator.pad_across_processes(local_ids_dev, dim=0, pad_index=-1)
            values_pad = accelerator.pad_across_processes(local_values_dev, dim=0)
            
            # Gather all (one sync)
            ids_all = accelerator.gather(ids_pad)
            values_all = accelerator.gather(values_pad)
            
            # Filter valid and update dataset
            valid = ids_all.ne(-1)
            idx_cpu = ids_all[valid].long().cpu()
            vals_cpu = values_all[valid].float().cpu()
            
            dataset.old_values[idx_cpu] = vals_cpu
        else:
            # Single process: just concatenate and update
            all_ids = torch.cat(local_ids_list, dim=0)
            all_values = torch.cat(local_values_list, dim=0)
            dataset.old_values[all_ids] = all_values

        accelerator.wait_for_everyone()
        getattr(value_model, 'module', value_model).train()


    #################################
    #             Inference         #
    #################################
    logger.info(f"***** Running inference (batch_size={batch_size_value_inference}) *****")

    compute_old_value_parallel(
        accelerator,
        dataset_lm,
        dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        )






    #################################
    #             Inference         #
    #################################
    logger.info("***** Calculate Advantage and Return *****")



    def compute_returns_and_advantages_from_fragments(
        dataset: TrainDataset,
        gamma: float,
        value_lam: float,          # For computing Return (Value model)
        policy_lam: float,         # For computing Advantage (Policy model)
        *,
        kl_in_reward_enabled: bool = False,
        kl_in_reward_beta: float = 0.01,
        atol: float = 1e-5
    ):
        """
        Read dataset.old_values (nonzero only at p_mask positions),
        aggregate fragments from the same original sequence to reconstruct the
        full token-level V^{old}, compute token-level R_j / A_j based on the
        per-sequence step_map / RLVR reward assignment, and finally write
        the R/A for the "currently trainable tokens" back into
        dataset.Return / dataset.adv (nonzero only at p_mask positions).

        [NEW BEHAVIOR - aligned with standard LLM PPO]:
        - Each sequence has only 1 row (N_rows = B)
        - p_mask includes all trainable positions (all non-padding response tokens)
        - Return/Advantage written to all trainable positions
        - Difference from old implementation: old used block-wise selection, only 1st token per step

        Return uses standard discounted return (R_t = r_t + gamma * R_{t+1}).
        Advantage uses policy_lam for GAE (Policy training signal).

        Supports both sequence-level reward (only last step) and step-level reward (all steps).
        Supports KL-in-reward: r'_t = r_t - beta * KL_t (VAPO).
        """
        L0, L1 = dataset.L0, dataset.L1
        B = dataset.step_map_all.size(0)
        N_rows = dataset.p_mask.size(0)

        # Pre-group the row indices belonging to each original sequence
        rows_by_seq = [[] for _ in range(B)]
        for row_idx in range(N_rows):
            s = int(dataset.seq_ids[row_idx].item())
            rows_by_seq[s].append(row_idx)

        # Clones for writing back later
        Return_mat = dataset.Return.clone()     # (N_rows, L)
        adv_mat    = dataset.adv.clone()
        old_vals   = dataset.old_values.clone() # (N_rows, L)

        # Process each original sequence
        for s in range(B):
            rows = rows_by_seq[s]
            if not rows:
                continue

            step_map_s = dataset.step_map_all[s].clone()          # (L1,)
            resp_ids_s = dataset.resp_input_ids_all[s].clone()    # (L1,)

            # Compute trainable mask (response segment):
            # not pad OR (pad but pad count <= post_num)
            is_pad = resp_ids_s.eq(dataset.pad_id)

            # Add EOS handling: exclude tokens after the first EOS
            eos_id = getattr(dataset, 'eos_id', None)
            eos_after_mask = torch.zeros_like(is_pad, dtype=torch.bool)
            if eos_id is not None:
                is_eos = resp_ids_s.eq(eos_id)
                if is_eos.any():
                    first_eos_pos = is_eos.float().argmax(dim=0).item()
                    eos_after_mask[first_eos_pos+1:] = True

            if dataset.post_num is None:
                trainable_mask = ~is_pad & ~eos_after_mask
            else:
                cum_pad = torch.cumsum(is_pad.int(), dim=0)
                trainable_mask = (~is_pad & ~eos_after_mask) | (is_pad & (cum_pad <= dataset.post_num) & ~eos_after_mask)

            # Find the maximum step id in the trainable region
            valid_steps = step_map_s[trainable_mask]
            assert valid_steps.numel() > 0, f"sequence {s}: no trainable tokens"
            last_step_id = int(valid_steps.max().item())

            # Token-level immediate reward r_j
            r_resp = torch.zeros(L1, dtype=torch.float32)
            
            # Step-level reward: assign reward to each step based on step_rewards
            if dataset.step_rewards_all is not None and len(dataset.step_rewards_all) > s:
                step_rewards_s = dataset.step_rewards_all[s]
                if step_rewards_s:  # Non-empty dict
                    for step_id, step_reward in step_rewards_s.items():
                        mask = (step_map_s == step_id) & trainable_mask
                        if mask.any():
                            r_resp[mask] = step_reward
                else:
                    # Fall back to sequence-level reward if step_rewards is empty
                    r_resp[(step_map_s == last_step_id) & trainable_mask] = dataset.per_seq_reward[s].item()
            else:
                # Sequence-level reward: only give reward to the last step
                r_resp[(step_map_s == last_step_id) & trainable_mask] = dataset.per_seq_reward[s].item()

            # KL-in-reward (VAPO): r'_t = r_t - beta * KL_t
            # IMPORTANT: KL penalty should only be applied at positions with reward,
            # NOT at every token. Otherwise, long sequences get penalized heavily.
            if kl_in_reward_enabled and dataset.kl_penalty is not None:
                # Step 1: Aggregate KL from fragment rows
                kl_resp = torch.zeros(L1, dtype=torch.float32)
                for row in rows_by_seq[s]:
                    pm = dataset.p_mask[row]
                    tail_mask = pm[L0:]
                    if not tail_mask.any():
                        continue
                    kl_row = dataset.kl_penalty[row, L0:L0+L1]
                    # Fill KL values (should not overlap due to p_mask design)
                    kl_resp[tail_mask] = kl_row[tail_mask]

                # Step 2: Apply KL penalty ONLY at reward positions
                # For sequence-level reward: only at the last step
                # For step-level reward: at each step with reward
                if dataset.step_rewards_all is not None and len(dataset.step_rewards_all) > s:
                    step_rewards_s = dataset.step_rewards_all[s]
                    if step_rewards_s:  # Non-empty dict: apply KL to each step with reward
                        for step_id, step_reward in step_rewards_s.items():
                            mask = (step_map_s == step_id) & trainable_mask
                            if mask.any():
                                # Average KL over this step's tokens, then apply penalty
                                kl_step_mean = kl_resp[mask].mean() if mask.any() else 0.0
                                r_resp[mask] -= kl_in_reward_beta * kl_step_mean
                    else:
                        # Fall back to sequence-level: only at last step
                        last_step_mask = (step_map_s == last_step_id) & trainable_mask
                        kl_last_mean = kl_resp[last_step_mask].mean() if last_step_mask.any() else 0.0
                        r_resp[last_step_mask] -= kl_in_reward_beta * kl_last_mean
                else:
                    # Sequence-level reward: only apply KL at the last step
                    last_step_mask = (step_map_s == last_step_id) & trainable_mask
                    kl_last_mean = kl_resp[last_step_mask].mean() if last_step_mask.any() else 0.0
                    r_resp[last_step_mask] -= kl_in_reward_beta * kl_last_mean

            # Aggregate token-level V^{old} from fragment rows
            V_resp = torch.zeros(L1, dtype=torch.float32)
            filled = torch.zeros(L1, dtype=torch.bool)
            union_mask_resp = torch.zeros(L1, dtype=torch.bool)

            for row in rows:
                pm = dataset.p_mask[row]  # (L0+L1,)
                # p_mask is always False in the first L0
                tail_mask = pm[L0:]       # (L1,)
                if not tail_mask.any():
                    continue
                vals_row = old_vals[row, L0:L0+L1]  # (L1,)
                # Each position should only be selected once
                assert not filled[tail_mask].any(), \
                    f"sequence {s}: duplicated selection in fragments"
                V_resp[tail_mask] = vals_row[tail_mask]
                filled[tail_mask] = True
                union_mask_resp |= tail_mask

            # All trainable positions must be covered
            assert torch.all(union_mask_resp[trainable_mask]), \
                f"sequence {s}: some trainable tokens lack value predictions"

            # Build an ordered list of step ids (trainable only)
            uniq_steps = torch.unique(step_map_s[trainable_mask], sorted=True)
            S = uniq_steps.numel()
            step_to_rank = {int(uniq_steps[i].item()): i for i in range(S)}

            # Per-step r_t^* / V_t^{*,old}
            r_star = torch.zeros(S, dtype=torch.float32)
            V_star = torch.zeros(S, dtype=torch.float32)

            for sid in uniq_steps.tolist():
                sid = int(sid)
                mask = (step_map_s == sid) & trainable_mask
                r_star[step_to_rank[sid]] = r_resp[mask].mean() if mask.any() else 0.0
                V_star[step_to_rank[sid]] = V_resp[mask].mean() if mask.any() else 0.0

            # Backward recursion for R_t^*
            R_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S-1, -1, -1):
                R_star[i] = r_star[i] + (gamma * R_star[i+1] if i+1 < S else 0.0)

            # TD residual and step-level GAE
            delta_star = torch.zeros(S, dtype=torch.float32)
            for i in range(S):
                v_next = V_star[i+1] if i+1 < S else 0.0
                delta_star[i] = r_star[i] - V_star[i] + gamma * v_next

            # ============ Compute Value_A (for Return) ============
            A_star_value = torch.zeros(S, dtype=torch.float32)
            for i in range(S-1, -1, -1):
                A_star_value[i] = delta_star[i] + (gamma * value_lam * A_star_value[i+1] if i+1 < S else 0.0)

            # ============ Compute Policy_A (for Advantage) ============
            A_star_policy = torch.zeros(S, dtype=torch.float32)
            for i in range(S-1, -1, -1):
                A_star_policy[i] = delta_star[i] + (gamma * policy_lam * A_star_policy[i+1] if i+1 < S else 0.0)

            # Map back to tokens: R_j, A_j
            R_resp = torch.zeros(L1, dtype=torch.float32)
            A_resp = torch.zeros(L1, dtype=torch.float32)
            for pos in torch.nonzero(trainable_mask, as_tuple=False).flatten().tolist():
                sid = int(step_map_s[pos].item())
                i = step_to_rank[sid]
                rj = r_resp[pos]
                R_next = R_star[i+1] if i+1 < S else 0.0
                V_next = V_star[i+1] if i+1 < S else 0.0
                A_value_next = A_star_value[i+1] if i+1 < S else 0.0
                A_policy_next = A_star_policy[i+1] if i+1 < S else 0.0

                # Return = token-level standard Return (does not depend on V, avoids circular dependency)
                R_resp[pos] = rj + gamma * R_next
                # Advantage uses policy_lam
                A_resp[pos] = (rj - V_resp[pos]) + gamma * V_next + gamma * policy_lam * A_policy_next

            # Write back into fragment rows (nonzero only at p_mask positions)
            R_full = torch.zeros(L0+L1, dtype=torch.float32)
            A_full = torch.zeros(L0+L1, dtype=torch.float32)
            R_full[L0:] = R_resp
            A_full[L0:] = A_resp

            for row in rows:
                pm = dataset.p_mask[row]
                Return_mat[row][pm] = R_full[pm]
                adv_mat[row][pm]    = A_full[pm]

            # Assertion 1: when gamma=value_lam=policy_lam=1,
            # For sequence-level reward: R_j equals the sequence reward (all tokens same)
            # For step-level reward: R_j for step i = sum of step_rewards from step i to last step
            # Note: A_j = R_j - V_j^{old} only when policy_lam=1, otherwise A uses different lambda
            if (abs(gamma - 1.0) < 1e-8 and abs(value_lam - 1.0) < 1e-8 and abs(policy_lam - 1.0) < 1e-8):
                # Compute expected Return for each token position
                expected_R_resp = torch.zeros(L1, dtype=torch.float32)

                if dataset.step_rewards_all is not None and len(dataset.step_rewards_all) > s:
                    step_rewards_s = dataset.step_rewards_all[s]
                    if step_rewards_s:
                        # For step-level reward: R for step i = sum of rewards from step i to last step
                        # Compute cumulative sum from each step to the end
                        for sid in uniq_steps.tolist():
                            i = step_to_rank[int(sid)]
                            # Sum of rewards from step i to last step
                            cumsum_from_i = sum(
                                step_rewards_s.get(int(uniq_steps[j].item()), 0.0)
                                for j in range(i, S)
                            )
                            mask = (step_map_s == int(sid)) & trainable_mask
                            expected_R_resp[mask] = cumsum_from_i
                    else:
                        # Fall back to sequence-level: all tokens get the same reward
                        expected_R_resp[trainable_mask] = dataset.per_seq_reward[s].item()
                else:
                    # Sequence-level reward: all tokens get the same reward
                    expected_R_resp[trainable_mask] = dataset.per_seq_reward[s].item()

                expected_R_full = torch.zeros(L0 + L1, dtype=torch.float32)
                expected_R_full[L0:] = expected_R_resp

                for row in rows:
                    pm = dataset.p_mask[row]
                    R_row = Return_mat[row][pm]
                    V_row = old_vals[row][pm]
                    A_row = adv_mat[row][pm]
                    R_expected = expected_R_full[pm]
                    assert torch.allclose(R_row, R_expected, atol=atol), \
                        f"gamma=value_lam=policy_lam=1 check failed (R) at seq {s}, row {row}"
                    # When policy_lam=1, A = R - V
                    assert torch.allclose(A_row, R_row - V_row, atol=atol), \
                        f"gamma=value_lam=policy_lam=1 check failed (A=R-V) at seq {s}, row {row}"

            # Assertion 2 is skipped since value_lam is typically 0.95 or 1.0, not 0

        # write back to data
        dataset.Return = Return_mat
        dataset.adv    = adv_mat


    gam = config.training.gam
    kl_in_reward_beta = OmegaConf.select(config, "kl_in_reward.beta", default=0.01)

    # Value lam: use config value for pretraining, 1.0 (MC) for main training
    # For computing Return (training target of Value model)
    value_pretrain_lam = OmegaConf.select(config, "training.value_pretrain_lam", default=None)
    if is_value_pretraining and value_pretrain_lam is not None:
        value_lam = value_pretrain_lam  # Pretraining: 0.95
    else:
        value_lam = 1.0  # Main training: MC

    # Policy lam: for computing Advantage (training signal for Policy)
    policy_lam = config.training.lam

    if accelerator.is_main_process and is_value_pretraining and value_pretrain_lam is not None:
        cprint(f"[Value Pretraining] value_lam={value_lam}", "cyan")

    compute_returns_and_advantages_from_fragments(
        dataset_lm, gam, value_lam, policy_lam,
        kl_in_reward_enabled=kl_in_reward_enabled,
        kl_in_reward_beta=kl_in_reward_beta,
        atol=1e-5
    )

    # Compute and log Value Model metrics
    if accelerator.is_main_process:
        try:
            value_metrics = collect_value_metrics_from_dataset(dataset_lm)
            # Output file path for value metrics
            value_results_file = f"./{project_name}/results/results-rl-.{project_name}.ckpt.optimized_value-{config.dataset.train_dataset}.txt"
            metrics_str = format_value_metrics(
                value_metrics, 
                epoch=config.experiment.current_epoch,
                output_file=value_results_file
            )
            logger.info("\n" + metrics_str)
        except Exception as e:
            logger.warning(f"Failed to compute value metrics: {e}")

    # Save metrics to temp file for pretrain convergence check (if in pretrain mode)
    if is_value_pretraining and accelerator.is_main_process and 'value_metrics' in locals():
        metrics_file = Path(project_name) / "temp_data" / "value_pretrain_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert torch tensors to python types for JSON serialization
        metrics_json = {k: float(v) if hasattr(v, 'item') else v for k, v in value_metrics.items()}
        with open(metrics_file, "w") as f:
            json.dump(metrics_json, f)
        logger.info(f"[Value Pretrain] Saved metrics to {metrics_file}")


    def save_dataset_tensors(dataset_lm, save_dir, name, accelerator, *,
                         start_pos: int, drop_num: int):
        from pathlib import Path, PurePath
        import time
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "extended_input_ids": dataset_lm.extended_input_ids,  # (N, L_ext)
            "p_mask":            dataset_lm.p_mask,               # (N, L)
            "tok_idx_ext":       dataset_lm.tok_idx_ext,          # (N, L)
            "labels":            dataset_lm.labels,               # (N, L)
            "adv":               dataset_lm.adv,                  # (N, L)
            "per_seq_reward":    dataset_lm.per_seq_reward,       # (B,) scalar reward for each original sequence
            "seq_ids":           dataset_lm.seq_ids,              # (N,) mapping from row to original sequence
            "correctness_all":   dataset_lm.correctness_all,      # (B,) bool, correctness for each original sequence
            "sample_idx_all":    dataset_lm.sample_idx_all,       # (N,) original sample indices for KL penalty mapping
            "resp_idx_all":      dataset_lm.resp_idx_all,         # (N,) response indices for KL penalty mapping
            # ===== Additional fields for train_policy_no_value.py =====
            "L0":                dataset_lm.L0,                   # Prompt length
            "L1":                dataset_lm.L1,                   # Response length
            "pad_id":            dataset_lm.pad_id,               # Pad token id
            "post_num":          dataset_lm.post_num,             # Post padding count
            "step_map_all":      dataset_lm.step_map_all,         # (B, L1) Step map for each original sequence
            "resp_input_ids_all": dataset_lm.resp_input_ids_all,  # (B, L1) Response token ids for each original sequence
            "step_rewards_all":  dataset_lm.step_rewards_all,     # List of dicts, per-step rewards
            # ===== Metadata =====
            "meta": {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_pos": int(start_pos),
                "drop_num":  int(drop_num),
            },
        }

        if accelerator.is_main_process:
            torch.save(payload, save_dir / f"{name}.pt")


    save_dataset_tensors(
        dataset_lm,                      #  extended_input_ids / p_mask / tok_idx_ext / labels / adv
        save_dir=Path(config.experiment.project) / "temp_data",
        name=f"{config.dataset.optimization_data}",  
        accelerator=accelerator,
        start_pos = start_pos,
        drop_num = drop_num
    )



    if config.experiment.current_epoch % config.experiment.train_value_every != 0:
        accelerator.wait_for_everyone()
        # Free GPU memory before NCCL cleanup to avoid OOM
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.end_training()
        return




    #################################
    #             Training          #
    #################################
    

    

    
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training / inference data = {input_ids_lm.shape[0]}")

    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_value}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()







    def forward_process(extended_input_ids, p_mask, tok_idx_ext, Return, old_values):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        # Dynamically create block_mask for this batch based on actual input size
        attention_mask = create_block_mask_for_batch(
            B=B, L=L, L0=L0,
            block_size=config.training.block_size,
            device=device,
            pad_id=pad_id,
            input_ids=extended_input_ids,
        )

        # Synchronize position_ids construction (following DiRL approach, consistent with policy training)
        # valid_indices: 1=prompt, 2=response, 0=padding
        valid_indices = torch.zeros(B, L, dtype=torch.long, device=device)
        valid_indices[:, :L0] = 1   # Set prompt portion to 1
        valid_indices[:, L0:L0+L1] = 2  # Set response portion to 2
        # Padding portion stays 0
        position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
        position_ids = torch.where(valid_indices == 0, torch.zeros_like(position_ids), position_ids)
        values = value_model(input_ids = extended_input_ids, attention_mask=attention_mask, position_ids = position_ids)
        values = values[:, :L0+L1]  # Keep original response positions
        # [NEW] No longer zero out values; p_mask includes all trainable positions (aligned with standard LLM PPO)
        # Old implementation: values = torch.where(p_mask, values, torch.zeros_like(values))

        # ===== Token-wise MSE Loss =====
        # Element-wise MSE: compute squared error at each trainable token, then average
        # This ensures per-token supervision signal (not sample-level aggregation)
        loss = ((values - Return) ** 2 * p_mask).sum() / p_mask.sum()

        return loss







    from tqdm.auto import tqdm

    # Record GPU memory before training loop
    log_gpu_memory(accelerator, "Before training loop")

    # Initialize metrics file for real-time logging (JSON Lines format)
    metrics_file = Path(project_name) / "results" / "training_metrics_value.jsonl"
    metrics_fp = None
    if accelerator.is_main_process:
        os.makedirs(metrics_file.parent, exist_ok=True)
        metrics_fp = open(metrics_file, "a")
        # Write config header
        config_header = {
            "type": "config",
            "data": {
                "current_epoch": config.experiment.current_epoch,
                "num_train_epochs": num_train_epochs,
                "batch_size": batch_size_value,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "total_batch_size": total_batch_size_lm,
                "learning_rate": config.optimizer.params.value_learning_rate,
                "num_training_data": p_mask.shape[0]
            }
        }
        metrics_fp.write(json.dumps(config_header) + "\n")
        metrics_fp.flush()

    loss_list = []
    global_step = 0
    total_batches = len(dataloader_lm)

    for epoch in range(first_epoch, num_train_epochs):

        getattr(value_model, 'module', value_model).train()

        # Accumulators for gradient accumulation steps
        acc_loss = 0.0
        acc_count = 0
        micro_step_count = 0  # Track micro steps for splitting large batches

        progress_bar = tqdm(
            dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            ncols=120,
            leave=True
        )


        for step, batch in enumerate(progress_bar, start=1):

            # Split large batch into smaller batches for training
            # DataLoader batch_size = batch_size_value_inference (e.g., 32)
            # Training batch_size = batch_size_value (e.g., 8)
            B = batch["extended_input_ids"].size(0)
            num_micro_batches = (B + batch_size_value - 1) // batch_size_value

            for micro_idx in range(num_micro_batches):
                start_idx = micro_idx * batch_size_value
                end_idx = min(start_idx + batch_size_value, B)

                # Extract micro batch
                micro_batch = {
                    "extended_input_ids": batch["extended_input_ids"][start_idx:end_idx].to(accelerator.device),
                    "p_mask": batch["p_mask"][start_idx:end_idx].to(accelerator.device),
                    "tok_idx_ext": batch["tok_idx_ext"][start_idx:end_idx].to(accelerator.device),
                    "Return": batch["Return"][start_idx:end_idx].to(accelerator.device),
                    "ids": batch["ids"][start_idx:end_idx],
                }

                data_time_m.update(time.time() - end)

                old_values = dataset_lm.old_values[micro_batch["ids"].cpu()].to(accelerator.device)

                loss_lm = forward_process(
                        extended_input_ids=micro_batch["extended_input_ids"],
                        p_mask=micro_batch["p_mask"],
                        tok_idx_ext=micro_batch["tok_idx_ext"],
                        Return=micro_batch["Return"],
                        old_values=old_values
                    )

                # Note: loss is already correctly normalized in forward_process:
                # loss = ((values - Return) ** 2 * p_mask).sum() / p_mask.sum()

                # Accumulate loss for metrics logging
                acc_loss += loss_lm.item()
                acc_count += 1

                accelerator.backward(loss_lm)

                micro_step_count += 1

                # Fix: force update after last batch
                is_last_batch = (step == total_batches) and (micro_idx == num_micro_batches - 1)
                should_update = ((micro_step_count + 1) % accelerator.gradient_accumulation_steps == 0) or is_last_batch

                if should_update:
                    # Clip gradients and get grad_norm
                    grad_norm = None
                    if config.training.max_grad_norm is not None:
                        grad_norm = accelerator.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    # Compute averaged metrics
                    avg_loss = acc_loss / acc_count if acc_count > 0 else 0.0
                    current_lr = lr_scheduler.get_last_lr()[0]
                    grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm if grad_norm is not None else 0.0)

                    # Log to console
                    logger.info(f"Step {global_step} (batch {step}/{total_batches}): loss={avg_loss:.6f}, grad_norm={grad_norm_value:.4f}, lr={current_lr:.2e}")

                    # Write step metrics to file
                    if accelerator.is_main_process and metrics_fp is not None:
                        step_data = {
                            "type": "step",
                            "data": {
                                "global_step": global_step,
                                "epoch": epoch + 1,
                                "batch_step": step,
                                "total_batches": total_batches,
                                "loss": avg_loss,
                                "grad_norm": grad_norm_value,
                                "lr": current_lr
                            }
                        }
                        metrics_fp.write(json.dumps(step_data) + "\n")
                        metrics_fp.flush()

                    # Reset accumulators
                    acc_loss = 0.0
                    acc_count = 0

                    torch.cuda.empty_cache()

                # Log loss to list
                loss_list.append(loss_lm.detach().float().cpu().item())

                # Clean up micro-batch tensors, free GPU memory
                del loss_lm
                del micro_batch
                torch.cuda.empty_cache()
            

                



        # Record epoch metrics to jsonl file
        if metrics_fp is not None and len(loss_list) > 0:
            epoch_avg_loss = sum(loss_list) / len(loss_list)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics = {
                "type": "epoch_metrics",
                "data": {
                    "epoch": epoch + 1,
                    "avg_loss": epoch_avg_loss,
                    "learning_rate": current_lr,
                    "num_steps": global_step
                }
            }
            metrics_fp.write(json.dumps(epoch_metrics) + "\n")
            metrics_fp.flush()

    # Clean up training loop variables, free GPU memory
    logger.info(f"[Rank {accelerator.process_index}] Cleaning up after training loop")
    del dataloader_lm
    del dataset_lm

    # Force GPU memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()

    # Record GPU memory after training loop
    log_gpu_memory(accelerator, "After training loop cleanup")

    accelerator.wait_for_everyone()

    # Move model to CPU before saving checkpoint to free GPU memory
    logger.info(f"[Rank {accelerator.process_index}] Moving value_model to CPU before saving checkpoint")
    value_model = value_model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Record GPU memory after moving model to CPU
    log_gpu_memory(accelerator, "After value_model moved to CPU")

    # save checkpoint at the end of training
    save_checkpoint(value_model, tokenizer, config, accelerator, config.model.optimized_value_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(value_model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}-value")

    # Close metrics file and write summary
    if accelerator.is_main_process and metrics_fp is not None:
        summary = {
            "type": "summary",
            "data": {
                "total_global_steps": global_step,
                "total_epochs": num_train_epochs,
                "training_completed": True
            }
        }
        metrics_fp.write(json.dumps(summary) + "\n")
        metrics_fp.close()
        logger.info(f"Training metrics saved to {metrics_file}")

    # Record training steps for pretrain progress tracking
    if accelerator.is_main_process:
        steps_file = Path(project_name) / "temp_data" / "value_train_steps.txt"
        steps_file.parent.mkdir(parents=True, exist_ok=True)
        with open(steps_file, "w") as f:
            f.write(str(global_step))

    # Free GPU memory before NCCL cleanup to avoid OOM during destroy_process_group
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    accelerator.end_training()



    if accelerator.is_main_process:

        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset

        def _mean(x): 
            return float(sum(x) / max(1, len(x))) 

        temp_len = 50
        first = loss_list[:temp_len]
        last  = loss_list[-temp_len:] if len(loss_list) >= temp_len else loss_list

        first_few_avg_loss = _mean(first)
        last_few_avg_loss  = _mean(last)
        avg_loss           = _mean(loss_list)

        output_text = (
            f"train step: {config.experiment.current_epoch}  "
            f"first_few_avg_loss: {first_few_avg_loss:.6f}  "
            f"last_few_avg_loss: {last_few_avg_loss:.6f}  "
            f"avg_loss: {avg_loss:.6f}  "
        )

        results_dir = Path(".") / project_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        outputs_result_name = results_dir / f"results-{outputs_name}.txt"

        cprint("\n\n" + output_text, color="green")
        with open(outputs_result_name, "a", encoding="utf-8", buffering=1) as f:
            f.write(output_text + "\n")
    




    
        








def save_checkpoint(model, tokenizer, config, accelerator, name):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        # Use save_pretrained directly without first getting state_dict to save GPU memory
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()

            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)  # e.g. .../modeling_sdar.py
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)

                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")





if __name__ == "__main__":
    main()




    
    


    