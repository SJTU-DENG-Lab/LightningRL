import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import contextlib
import json
import logging
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models import SDARForCausalLM
from models.logging import set_verbosity_error, set_verbosity_info
from models.lr_schedulers import get_scheduler
from train.prompting_utils import UniversalPrompting

SYSTEM_PROMPT_LEN = 28

from train.utils import AverageMeter, flatten_omega_conf, get_config

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
            logger.info(
                f"[GPU {i}] {message}: "
                f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                f"Max={max_allocated:.2f}GB, Total={total:.2f}GB"
            )


# ===== Torch Compile Cache Clear for Checkpoint Loading =====
def clear_torch_compile_cache(logger):
    """
    Clear torch.compile compilation cache after loading model from checkpoint.

    When loading a model from checkpoint (epoch > 1), the torch.compile
    cache (especially for functions with @torch.compile decorator like fused_flex_attention)
    may contain kernels optimized for the old weights, causing CUDA illegal
    memory access during backward pass with gradient checkpointing.

    This function clears the compilation cache to force recompilation with new weights.
    """
    try:
        # Preferred: public API (PyTorch 2.5+)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
            logger.info("Using torch.compiler.reset() to clear compilation cache")
        # Fallback: internal API
        elif hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
            logger.info("Using torch._dynamo.reset() to clear compilation cache")
        else:
            logger.warning("No torch.compile cache clearing method available")
    except Exception as e:
        logger.warning(f"Failed to clear torch.compile cache: {e}")


# ===== End Torch Compile Cache Clear =====


class TrainDataset(Dataset):
    def __init__(
        self, extended_input_ids, p_mask, tok_idx_ext, labels, adv, correctness=None, seq_ids=None, ref_logprobs=None
    ):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.adv = adv
        self.logp_old_tok = torch.full((len(extended_input_ids), p_mask.shape[1]), float("-inf"))
        # Correctness for NLL loss (VAPO): map from row to original sequence correctness
        # correctness is (B,) bool tensor, seq_ids is (N,) mapping from row to seq
        if correctness is not None and seq_ids is not None:
            # Map correctness from (B,) to (N,) using seq_ids
            self.correctness = correctness[seq_ids]  # (N,)
        else:
            self.correctness = None

        # Reference model logprobs for KL penalty
        self.ref_logprobs = ref_logprobs  # (N, L) or None

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        corr = self.correctness[idx] if self.correctness is not None else True

        # Handle ref_logprobs
        if self.ref_logprobs is not None:
            ref_logprob = self.ref_logprobs[idx]
        else:
            ref_logprob = torch.tensor([])  # Empty tensor to avoid collate errors

        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.adv[idx],
            corr,
            ref_logprob,  # Add ref_logprob to return tuple
        )


@contextlib.contextmanager
def disable_gradient_checkpointing(model):
    """
    Temporarily disable gradient checkpointing for the model and all submodules.

    Purpose: avoid gradient checkpointing compatibility issues with no_grad
             when computing logits in a no_grad context.

    Usage:
        with disable_gradient_checkpointing(model):
            outputs = model(...)
    """
    original_states = {}

    # Save original state and disable
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            original_states[name] = module.gradient_checkpointing
            module.gradient_checkpointing = False

    try:
        yield
    finally:
        # Restore original state
        for name, module in model.named_modules():
            if name in original_states:
                module.gradient_checkpointing = original_states[name]


@torch.no_grad()
def compute_model_logprobs(
    model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, batch_size, accelerator, model_name="model"
):
    """
    General function: compute log probabilities of a model over all data.

    Args:
        model: model instance (policy or ref_model)
        extended_input_ids: (N, L) token ids
        p_mask: (N, L) bool mask for masked positions
        tok_idx_ext: (N, L) token indices
        labels: (N, L) target labels
        start_pos: int, prompt length
        batch_size: int, batch size for computation
        accelerator: Accelerator instance
        model_name: str, model name for logging

    Returns:
        all_logprobs: (N, L) log probabilities on CPU
    """
    logger.info(f"Computing {model_name} logprobs...")

    # Create temporary dataset and dataloader
    temp_dataset = torch.utils.data.TensorDataset(extended_input_ids, p_mask, tok_idx_ext, labels)
    temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    # Set model to train mode (to support masked_indices), but use no_grad
    model.train()

    logprobs_list = []
    for batch_data in temp_loader:
        b_input_ids, b_p_mask, b_tok_idx_ext, b_labels = batch_data
        B, L = b_input_ids.shape
        device = accelerator.device

        # Move to GPU
        b_input_ids = b_input_ids.to(device)
        b_p_mask = b_p_mask.to(device)
        b_tok_idx_ext = b_tok_idx_ext.to(device)
        b_labels = b_labels.to(device)

        # Construct valid_indices
        valid_indices = torch.zeros(B, L, dtype=torch.long, device=device)
        valid_indices[:, :start_pos] = 1  # prompt
        valid_indices[:, start_pos:] = 2  # response

        # Construct position_ids
        position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
        position_ids = torch.where(valid_indices == 0, torch.zeros_like(position_ids), position_ids)

        # Model forward with return_logits=True
        # ===== Disable gradient checkpointing to speed up ref logp computation =====
        # Align with compute_logp_old_tok_parallel, avoid compatibility issue with no_grad
        with disable_gradient_checkpointing(model):
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=None,
                position_ids=position_ids,
                labels=b_labels,
                masked_indices=b_p_mask,
                return_logits=True,
            )
        # ===== End disable gradient checkpointing =====

        # Compute log probabilities
        logits = outputs.logits  # (M, V) where M = sum(b_p_mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create full logp tensor (B, L)
        logp_batch = torch.zeros(B, L, device=device, dtype=logits.dtype)
        labels_masked = b_labels[b_p_mask]
        logp_masked = log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)
        logp_batch[b_p_mask] = logp_masked

        # Save to CPU
        logprobs_list.append(logp_batch.cpu())

        # Cleanup (removed frequent empty_cache to improve performance)
        del outputs, logits, log_probs, logp_masked, labels_masked, logp_batch
        del b_input_ids, b_p_mask, b_tok_idx_ext, b_labels, position_ids, valid_indices
        # Removed: torch.cuda.empty_cache()  # Frequent calls block; unified cleanup after loop instead

    # Concatenate all batches
    all_logprobs = torch.cat(logprobs_list, dim=0)
    del logprobs_list, temp_dataset, temp_loader
    torch.cuda.empty_cache()

    logger.info(f"{model_name} logprobs computed: shape {all_logprobs.shape}")
    return all_logprobs


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Declare ref_model for KL divergence
    ref_model = None

    project_name = config.experiment.project
    if config.experiment.current_epoch <= 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_name

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
    # Make one log on every process with the configuration for debugging.
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

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_prompt_len=config.training.max_prompt_len,
        max_gen_length=config.training.max_gen_length,
        ignore_id=-100,
    )

    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    model = SDARForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    # ===== Fix: clear torch.compile cache after loading checkpoint =====
    # When loading model from checkpoint (epoch > 1), clear torch.compile compilation cache
    # Especially important for functions decorated with @torch.compile (e.g. fused_flex_attention)
    if config.experiment.current_epoch > 1:
        logger.info("=" * 80)
        logger.info("Loaded model from checkpoint, clearing torch.compile cache")
        logger.info("=" * 80)
        clear_torch_compile_cache(logger)
        logger.info("torch.compile cache cleared, will recompile with new weights")
        logger.info("=" * 80)
    # ===== End fix =====

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False

    # ===== Load Reference Model for KL Divergence =====
    if config.training.kl_beta > 0:
        logger.info("=" * 80)
        logger.info("Loading reference model for KL divergence")
        logger.info("=" * 80)

        ref_model_path = config.model.pretrained_model
        logger.info(f"Loading reference model from {ref_model_path}")
        ref_model = SDARForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True, torch_dtype="auto")
        if hasattr(ref_model, "config"):
            ref_model.config.fuse_cross_entropy = False
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        logger.info("Reference model loaded and frozen")

        # Keep ref_model on CPU to save GPU memory
        ref_model = ref_model.cpu()
        logger.info("Reference model kept on CPU")
    else:
        logger.info("KL penalty disabled (kl_beta=0), skipping reference model loading")

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    # Add warning (no auto-fix, since data is loaded from pt files)
    if pad_id == eos_id:
        logger.warning("=" * 80)
        logger.warning("WARNING: pad_token_id == eos_token_id detected!")
        logger.warning(f"pad_token_id = {pad_id}, eos_token_id = {eos_id}")
        logger.warning("This may cause issues if EOS tokens exist in the data.")
        logger.warning("Please check the data generation in train_policy_no_value.py")
        logger.warning("=" * 80)

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.policy_learning_rate,
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
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, adv, corr, ref_logp = zip(*batch)

        # Handle ref_logp: check if all are empty tensors
        ref_logp_list = list(ref_logp)
        if all(r.numel() == 0 for r in ref_logp_list):
            ref_logp_stacked = None
        else:
            ref_logp_stacked = torch.stack(ref_logp_list)

        return {
            "ids": torch.tensor(idx),
            "extended_input_ids": torch.stack(extended_input_ids),
            "p_mask": torch.stack(p_mask),
            "tok_idx_ext": torch.stack(tok_idx_ext),
            "labels": torch.stack(labels),
            "adv": torch.stack(adv),
            "correctness": torch.tensor(corr, dtype=torch.bool),
            "ref_logp": ref_logp_stacked,  # Add ref_logp to batch dict
        }

    dataset_load = torch.load(
        Path(project_name) / "temp_data" / f"{config.dataset.optimization_data}.pt", map_location="cpu"
    )
    extended_input_ids = dataset_load["extended_input_ids"]
    p_mask = dataset_load["p_mask"]
    tok_idx_ext = dataset_load["tok_idx_ext"]
    labels = dataset_load["labels"]
    adv = dataset_load["adv"]
    per_seq_reward = dataset_load.get("per_seq_reward", None)  # (B,) scalar reward for each original sequence
    seq_ids = dataset_load.get("seq_ids", None)  # (N,) mapping from row to original sequence
    correctness_all = dataset_load.get("correctness_all", None)  # (B,) bool, correctness for NLL loss
    start_pos = dataset_load["meta"]["start_pos"]
    drop_num = dataset_load["meta"]["drop_num"]

    _, L = p_mask.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.post_num

    # Use batch_size_policy if specified, otherwise fall back to batch_size_lm for compatibility
    batch_size_policy = OmegaConf.select(
        config, "training.batch_size_policy", default=OmegaConf.select(config, "training.batch_size_lm", default=8)
    )

    #################################
    # Compute Reference Logprobs    #
    #################################
    ref_logprobs = None
    if config.training.kl_beta > 0 and ref_model is not None:
        logger.info("=" * 80)
        logger.info("Computing reference model logprobs")
        logger.info("=" * 80)

        time_ref_start = time.time()

        # Move ref_model to GPU
        logger.info("Moving reference model to GPU")
        ref_model = ref_model.to(accelerator.device)

        # Compute ref logprobs
        ref_logprobs = compute_model_logprobs(
            model=ref_model,
            extended_input_ids=extended_input_ids,
            p_mask=p_mask,
            tok_idx_ext=tok_idx_ext,
            labels=labels,
            start_pos=start_pos,
            batch_size=batch_size_policy,
            accelerator=accelerator,
            model_name="reference",
        )

        time_ref = time.time() - time_ref_start
        logger.info(f"Reference logprobs computed in {time_ref:.2f}s")

        # Move ref_model back to CPU and cleanup
        ref_model.eval()
        ref_model = ref_model.cpu()
        torch.cuda.empty_cache()
        import gc

        gc.collect()

        accelerator.wait_for_everyone()
        logger.info("Reference model moved back to CPU")
    else:
        logger.info("Skipping reference logprobs computation (kl_beta=0 or ref_model=None)")

    # basic_block_attention and process_pad have been removed
    # The model internally builds the BlockMask required for Block Diffusion
    # Reason: data length is L but function expects L0 + 2*L1, causing dimension mismatch

    dataset_lm = TrainDataset(
        extended_input_ids,
        p_mask,
        tok_idx_ext,
        labels,
        adv,
        correctness=correctness_all,
        seq_ids=seq_ids,
        ref_logprobs=ref_logprobs,
    )

    total_batch_size_lm = batch_size_policy * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    train_dataloader_lm = DataLoader(
        dataset_lm, batch_size=batch_size_policy, sampler=None, collate_fn=simple_collate, num_workers=0
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )
    # Ensure SDARForCausalLM model is in training mode
    # DeepSpeed direct assignment requires explicitly setting training state of inner modules
    getattr(model, "module", model).train()
    import torch.nn.functional as F

    @torch.no_grad()
    def compute_logp_old_tok_parallel(accelerator, dataset, train_dataloader_lm, start_pos, pad_id, batch_size):
        # Training mode ensures the training branch is taken, returning logits with the correct shape
        # DeepSpeed direct assignment requires explicitly setting training state of inner modules
        getattr(model, "module", model).train()

        dl = train_dataloader_lm

        for batch in dl:
            ids = batch["ids"]  # (b,)
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            B, L = p_mask.shape
            L0 = start_pos
            L1 = L - L0
            device = extended_input_ids.device

            # Removed basic_block_attention and process_pad; use attention_mask=None
            # The model internally builds the required BlockDiffusion BlockMask
            position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
            position_ids = torch.where(tok_idx_ext > 0, position_ids, torch.zeros_like(position_ids))

            # Using masked_indices parameter, model only returns logits at masked positions (M, V)
            # Where M = sum(p_mask), greatly reducing computation
            # Temporarily disable gradient checkpointing to avoid conflict with no_grad
            with disable_gradient_checkpointing(model):
                logits = model(
                    input_ids=extended_input_ids,
                    attention_mask=None,  # ← removed basic_block_attention
                    position_ids=position_ids,
                    labels=labels,  # ← added labels parameter to ensure correct prompt_mask computation
                    masked_indices=p_mask,  # use p_mask as masked_indices
                    return_logits=True,  # return logits for logp computation
                ).logits  # (M, V), M = sum(p_mask)

            # logits are already in masked shape, compute log_softmax directly
            log_probs = F.log_softmax(logits, dim=-1)  # (M, V)

            # Extract labels at corresponding positions and compute logp
            labels_masked = labels[p_mask]  # (M,)
            logp_masked = log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)  # (M,)

            # Fill back to full shape (B, L)
            logp_tok = torch.zeros(B, L, device=device, dtype=log_probs.dtype)
            logp_tok[p_mask] = logp_masked

            # Store back to dataset (must be on CPU)
            dataset.logp_old_tok[ids] = logp_tok.float().cpu()

        accelerator.wait_for_everyone()

        model.train()

    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_logp_old_tok_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        batch_size=batch_size_policy,
    )

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")

    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {p_mask.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_policy}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    def forward_process(
        extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok, correctness=None, logp_ref_tok=None
    ):
        B, L = p_mask.shape
        L0 = start_pos
        L1 = L - L0
        device = extended_input_ids.device

        # Removed basic_block_attention and process_pad; use attention_mask=None
        # The model automatically calls prepare_for_bd_training to build the required BlockMask
        # Fix: use labels and start_pos to build valid_indices for correct padding identification
        # valid_indices: 1=prompt, 2=response, 0=padding
        valid_indices = torch.zeros(B, L, dtype=torch.long, device=device)
        valid_indices[:, :start_pos] = 1  # Set prompt portion to 1
        valid_indices[:, start_pos:] = 2  # Set response portion to 2
        # Padding portion stays 0 (initialized as 0)

        position_ids = torch.arange(L, device=device).long().unsqueeze(0).expand(B, -1)
        position_ids = torch.where(valid_indices == 0, torch.zeros_like(position_ids), position_ids)

        # Prepare is_real: mark non-padding samples as real
        valid_mask = labels != -100
        is_real = valid_mask.any(dim=1)  # (B,) bool

        # Get configuration parameters
        eps = config.training.eps
        eps_high = OmegaConf.select(config, "training.eps_high", default=eps + 0.08)
        nll_weight = OmegaConf.select(config, "training.nll_loss_weight", default=0.0)
        kl_beta = OmegaConf.select(config, "training.kl_beta", default=0.0)

        # Support string-form kl_estimator config ("k1", "k2", "k3")
        kl_estimator = OmegaConf.select(config, "training.kl_estimator", default="k3")
        # Validate
        if kl_estimator not in ["k1", "k2", "k3"]:
            logger.warning(f"Invalid kl_estimator '{kl_estimator}', using default 'k3'")
            kl_estimator = "k3"
        logger.info(f"Using KL estimator: {kl_estimator.upper()}")

        # Read three independent reduction modes
        policy_reduction_mode = OmegaConf.select(config, "training.policy_reduction_mode", default="token")
        kl_reduction_mode = OmegaConf.select(config, "training.kl_reduction_mode", default="token")
        nll_reduction_mode = OmegaConf.select(config, "training.nll_reduction_mode", default="token")

        # Ensure inner SDARForCausalLM model is in training mode
        # DeepSpeed direct assignment requires explicitly setting training state of inner modules
        getattr(model, "module", model).train()

        # Call the model's compute_rl_loss branch
        # Note: the model automatically handles Block Diffusion expansion (input_ids -> concat_inputs_ids)
        outputs = model(
            input_ids=extended_input_ids,
            attention_mask=None,  # ← model internally builds BlockMask
            position_ids=position_ids,
            labels=labels,
            masked_indices=p_mask,  # ← p_mask as masked_indices
            compute_rl_loss=True,  # ← use RL loss branch
            rl_p_mask=p_mask,  # ← p_mask as rl_p_mask
            rl_adv=adv,
            rl_is_real=is_real,  # ← mark real samples
            rl_logp_old_tok=logp_old_tok,
            rl_logp_ref_tok=logp_ref_tok,  # ← Add this line
            rl_ppo_eps=eps,
            rl_ppo_eps_high=eps_high,
            rl_correctness=correctness if correctness is not None else None,
            rl_nll_weight=nll_weight,
            rl_kl_beta=kl_beta,  # ← KL penalty coefficient
            rl_kl_estimator=kl_estimator,  # ← KL estimator type ("k1", "k2", "k3")
            rl_return_entropy=True,
            loss_mean=False,
            # Three independent reduction modes
            policy_reduction_mode=policy_reduction_mode,
            kl_reduction_mode=kl_reduction_mode,
            nll_reduction_mode=nll_reduction_mode,
        )

        # Extract return values
        # outputs.loss = policy_loss + kl_loss + nll_loss (total loss)
        total_loss = outputs.loss
        kl_loss = outputs.kl_loss if hasattr(outputs, "kl_loss") else torch.tensor(0.0, device=device)
        nll_loss = outputs.nll_loss if hasattr(outputs, "nll_loss") else torch.tensor(0.0, device=device)
        ratio_mean = outputs.ratio_mean if hasattr(outputs, "ratio_mean") else torch.tensor(0.0, device=device)
        clip_frac = outputs.clip_frac if hasattr(outputs, "clip_frac") else torch.tensor(0.0, device=device)

        # Separate pure policy_loss from total_loss
        policy_loss = total_loss - kl_loss - nll_loss

        # Compute metrics
        kl_mean = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        nll_mean = nll_loss.item() if isinstance(nll_loss, torch.Tensor) else nll_loss
        ratio_mean_val = ratio_mean.item() if isinstance(ratio_mean, torch.Tensor) else ratio_mean
        clip_frac_val = clip_frac.item() if isinstance(clip_frac, torch.Tensor) else clip_frac

        return {
            "total_loss": total_loss,  # Use total loss returned by model directly
            "policy_loss": policy_loss.item(),  # Pure policy_loss
            "nll_loss": nll_mean,  # Actual nll_loss
            "ratio_mean": ratio_mean_val,
            "clip_frac": clip_frac_val,
            "kl_mean": kl_mean,
            "entropy": outputs.entropy.item() if hasattr(outputs, "entropy") else 0.0,
        }

    from tqdm.auto import tqdm

    # Initialize metrics file for real-time logging (JSON Lines format)
    # All epochs append to the same file
    metrics_file = Path(project_name) / "results" / "training_metrics.jsonl"
    metrics_fp = None
    if accelerator.is_main_process:
        os.makedirs(metrics_file.parent, exist_ok=True)
        # Check if file exists to determine if we need to write header
        file_exists = metrics_file.exists()
        # Use append mode "a" so all epochs append to the same file
        metrics_fp = open(metrics_file, "a")

        # Write config header only for new file (first time)
        if not file_exists:
            config_header = {
                "type": "config",
                "data": {
                    "project": config.experiment.project,
                    "batch_size": batch_size_policy,
                    "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                    "total_batch_size": total_batch_size_lm,
                    "eps": config.training.eps,
                    "learning_rate": config.optimizer.params.policy_learning_rate,
                },
            }
            metrics_fp.write(json.dumps(config_header) + "\n")
            metrics_fp.flush()

    global_step = 0
    # Accumulators for gradient accumulation steps
    acc_policy_loss = 0.0
    acc_nll_loss = 0.0
    acc_ratio = 0.0
    acc_clip_frac = 0.0
    acc_kl = 0.0
    acc_entropy = 0.0
    acc_reward_sum = 0.0
    acc_reward_sq_sum = 0.0
    acc_reward_count = 0
    acc_correct_count = 0
    acc_total_count = 0
    acc_total_reward_sum = 0.0
    acc_total_reward_sq_sum = 0.0
    acc_total_reward_count = 0
    acc_acc_reward_sum = 0.0
    acc_speed_reward_sum = 0.0

    # Record GPU memory before training loop
    log_gpu_memory(accelerator, "Before training loop")

    total_batches = len(train_dataloader_lm)

    for epoch in range(first_epoch, num_train_epochs):
        model.train()

        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(progress_bar, start=1):
            # for loss calculation

            data_time_m.update(time.time() - end)

            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            adv = batch["adv"].to(accelerator.device)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)
            correctness = batch["correctness"].to(accelerator.device)
            ref_lp = (
                batch["ref_logp"].to(accelerator.device) if batch["ref_logp"] is not None else None
            )  # Add this line

            if torch.isneginf(old_lp).any().item():
                print(old_lp)

            # Collect reward stats (using actual per-sequence rewards, not advantages)
            if per_seq_reward is not None and seq_ids is not None and correctness_all is not None:
                batch_ids = batch["ids"].cpu()
                batch_seq_ids = seq_ids[batch_ids]  # (batch_size,) sequence ids for each row in batch
                batch_total_rewards = per_seq_reward[batch_seq_ids]  # (batch_size,) total rewards
                batch_correctness = correctness_all[batch_seq_ids]  # (batch_size,) correctness

                # Calculate acc_reward and speed_reward
                # If correct: total_reward = 1.0 + normalized_tpf, so acc_reward = 1.0, speed_reward = total_reward - 1.0
                # If wrong: total_reward = -1.0 + normalized_tpf, so acc_reward = -1.0, speed_reward = total_reward + 1.0
                batch_acc_rewards = torch.where(
                    batch_correctness, torch.ones_like(batch_total_rewards), -torch.ones_like(batch_total_rewards)
                )
                batch_speed_rewards = torch.where(
                    batch_correctness, batch_total_rewards - 1.0, batch_total_rewards + 1.0
                )

                # Accumulate total_reward (for mean/std calculation)
                acc_total_reward_sum += batch_total_rewards.sum().item()
                acc_total_reward_sq_sum += (batch_total_rewards**2).sum().item()
                acc_total_reward_count += batch_total_rewards.numel()

                # Accumulate acc_reward and speed_reward (for mean calculation)
                acc_acc_reward_sum += batch_acc_rewards.sum().item()
                acc_speed_reward_sum += batch_speed_rewards.sum().item()

                # Keep old reward accumulation for backward compatibility
                acc_reward_sum += batch_total_rewards.sum().item()
                acc_reward_sq_sum += (batch_total_rewards**2).sum().item()
                acc_reward_count += batch_total_rewards.numel()

            result = forward_process(
                extended_input_ids=extended_input_ids,
                p_mask=p_mask,
                tok_idx_ext=tok_idx_ext,
                labels=labels,
                adv=adv,
                logp_old_tok=old_lp,
                correctness=correctness,
                logp_ref_tok=ref_lp,  # Add this line
            )

            # Use total_loss returned by model (already per-token averaged, no extra normalization needed)
            # Formula: L = (1 / sum_tokens) * sum(surrogate_p)
            # Model uses .mean() internally, use directly
            accelerator.backward(result["total_loss"])

            # Accumulate metrics for this gradient accumulation window
            acc_policy_loss += result["policy_loss"]
            acc_nll_loss += result["nll_loss"]
            acc_ratio += result["ratio_mean"]
            acc_clip_frac += result["clip_frac"]
            acc_kl += result["kl_mean"]
            acc_entropy += result.get("entropy", 0.0)
            acc_correct_count += correctness.sum().item()
            acc_total_count += correctness.numel()

            if step < 10:
                print(result["total_loss"])

            # Clean up tensors from this batch, free GPU memory
            del result
            del extended_input_ids, p_mask, tok_idx_ext, labels, adv, old_lp, correctness
            torch.cuda.empty_cache()

            # Fix: force update after last batch
            is_last_batch = step == total_batches
            should_update = ((step + 1) % accelerator.gradient_accumulation_steps == 0) or is_last_batch

            if should_update:
                # Clip gradients and get grad_norm
                grad_norm = None
                if config.training.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Compute averaged metrics over gradient accumulation steps
                n_acc = accelerator.gradient_accumulation_steps
                avg_policy_loss = acc_policy_loss / n_acc
                avg_nll_loss = acc_nll_loss / n_acc
                avg_ratio = acc_ratio / n_acc
                avg_clip_frac = acc_clip_frac / n_acc
                avg_kl = acc_kl / n_acc
                avg_entropy = acc_entropy / n_acc
                grad_norm_value = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else (grad_norm if grad_norm is not None else 0.0)
                )

                # Compute reward mean and std
                if acc_reward_count > 0:
                    reward_mean = acc_reward_sum / acc_reward_count
                    reward_var = (acc_reward_sq_sum / acc_reward_count) - (reward_mean**2)
                    reward_std = math.sqrt(max(reward_var, 0))
                else:
                    reward_mean = 0.0
                    reward_std = 0.0

                # Compute new reward metrics (total_reward, acc_reward, speed_reward)
                if acc_total_reward_count > 0:
                    avg_total_reward = acc_total_reward_sum / acc_total_reward_count
                    total_reward_var = (acc_total_reward_sq_sum / acc_total_reward_count) - (avg_total_reward**2)
                    total_reward_std = math.sqrt(max(total_reward_var, 0))
                    avg_acc_reward = acc_acc_reward_sum / acc_total_reward_count
                    avg_speed_reward = acc_speed_reward_sum / acc_total_reward_count
                else:
                    avg_total_reward = 0.0
                    total_reward_std = 0.0
                    avg_acc_reward = 0.0
                    avg_speed_reward = 0.0

                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0]

                # Write step metrics to file
                nll_loss_weight = OmegaConf.select(config, "training.nll_loss_weight", default=0.0)
                correct_ratio = acc_correct_count / acc_total_count if acc_total_count > 0 else 0.0
                if accelerator.is_main_process and metrics_fp is not None:
                    step_data = {
                        "type": "step",
                        "data": {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "batch_step": step,
                            "reward_mean": reward_mean,
                            "reward_std": reward_std,
                            "total_reward": avg_total_reward,
                            "acc_reward": avg_acc_reward,
                            "speed_reward": avg_speed_reward,
                            "entropy": avg_entropy,
                            "policy_loss": avg_policy_loss,
                            "nll_loss": avg_nll_loss,
                            "total_loss": avg_policy_loss + nll_loss_weight * avg_nll_loss,
                            "ratio_mean": avg_ratio,
                            "clip_frac": avg_clip_frac,
                            "correct_ratio": correct_ratio,
                            "grad_norm": grad_norm_value,
                            "kl": avg_kl,
                            "lr": current_lr,
                        },
                    }
                    metrics_fp.write(json.dumps(step_data) + "\n")
                    metrics_fp.flush()

                # Reset accumulators
                acc_policy_loss = 0.0
                acc_nll_loss = 0.0
                acc_ratio = 0.0
                acc_clip_frac = 0.0
                acc_kl = 0.0
                acc_entropy = 0.0
                acc_reward_sum = 0.0
                acc_reward_sq_sum = 0.0
                acc_reward_count = 0
                acc_correct_count = 0
                acc_total_count = 0
                acc_total_reward_sum = 0.0
                acc_total_reward_sq_sum = 0.0
                acc_total_reward_count = 0
                acc_acc_reward_sum = 0.0
                acc_speed_reward_sum = 0.0

                torch.cuda.empty_cache()

    # Clean up training loop variables, free GPU memory
    logger.info(f"[Rank {accelerator.process_index}] Cleaning up after training loop")
    del train_dataloader_lm
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
    logger.info(f"[Rank {accelerator.process_index}] Moving model to CPU before saving checkpoint")
    model = model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Record GPU memory after moving model to CPU
    log_gpu_memory(accelerator, "After model moved to CPU")

    # Close metrics file and write summary
    if accelerator.is_main_process and metrics_fp is not None:
        summary = {
            "type": "summary",
            "data": {"total_global_steps": global_step, "total_epochs": num_train_epochs, "training_completed": True},
        }
        metrics_fp.write(json.dumps(summary) + "\n")
        metrics_fp.close()
        logger.info(f"Training metrics saved to {metrics_file}")

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}-policy")

    # Free GPU memory before NCCL cleanup to avoid OOM during destroy_process_group
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name):
    import glob
    import importlib
    import inspect
    import json
    import os
    import time
    from pathlib import Path

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
