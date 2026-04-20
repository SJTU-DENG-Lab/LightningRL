# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED


class UnmaskingProcessor:

    def __init__(self, dllm_config: DLLMConfig):
        self.dllm_config = dllm_config
        # Track which blocks should force static decoding (key: batch_idx)
        self.force_static_mask = None

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def _get_denoise_num(self):
        """Get denoise num."""
        block_size = self.dllm_config.block_length
        denoising_steps = self.dllm_config.denoising_steps
        if denoising_steps is None:
            denoising_steps = block_size
        num = block_size // self.dllm_config.denoising_steps
        num = max(1, min(num, block_size))
        return num

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """static."""
        block_size = self.dllm_config.block_length
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(topk, dim=-1)
        dllm_unmasked = dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)
        return dllm_mask.flatten()

    def low_confidence_dynamic(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """dynamic."""
        block_size = self.dllm_config.block_length
        threshold = self.dllm_config.confidence_threshold
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(1, dim=-1)
        scores = scores.scatter(-1, indices, threshold)

        is_masked = is_masked.view_as(dllm_mask)
        is_masked &= scores >= threshold
        dllm_mask[is_masked] = DLLM_UNMASKED
        return dllm_mask.flatten()
    
    def low_confidence_dynamic_enhanced(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """Enhanced dynamic strategy: if max score < threshold in current block, switch to static for remaining iterations."""
        block_size = self.dllm_config.block_length
        threshold = self.dllm_config.confidence_threshold
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        
        # Initialize force_static_mask if needed
        num_blocks = scores.shape[0]
        if self.force_static_mask is None or self.force_static_mask.shape[0] != num_blocks:
            self.force_static_mask = torch.zeros(num_blocks, dtype=torch.bool, device=scores.device)
        
        # Find blocks that should use static decoding (max score < threshold)
        max_scores_per_block = scores.max(dim=-1)[0]
        forced_blocks = max_scores_per_block < threshold
        self.force_static_mask |= forced_blocks
        
        # For blocks marked as force_static, use static strategy
        topk = self._get_denoise_num()
        is_masked_view = is_masked.view(-1, block_size)
        
        for block_idx in range(num_blocks):
            if self.force_static_mask[block_idx]:
                # Use static decoding for this block
                block_scores = scores[block_idx]
                block_mask = dllm_mask[block_idx]
                block_is_masked = is_masked_view[block_idx]
                
                _, indices = block_scores.topk(topk, dim=-1)
                dllm_unmasked = block_mask.scatter(-1, indices, DLLM_UNMASKED)
                dllm_mask[block_idx] = torch.where(block_is_masked, dllm_unmasked, block_mask)
            else:
                # Use dynamic decoding for this block
                block_scores = scores[block_idx]
                block_mask = dllm_mask[block_idx]
                block_is_masked = is_masked_view[block_idx]
                
                _, indices = block_scores.topk(1, dim=-1)
                block_scores = block_scores.scatter(-1, indices, threshold)
                block_is_masked_updated = block_is_masked & (block_scores >= threshold)
                dllm_mask[block_idx][block_is_masked_updated] = DLLM_UNMASKED
        
        return dllm_mask.flatten()

    def sequential(self, dllm_mask: torch.Tensor):
        """sequential."""
        block_size = self.dllm_config.block_length
        denoise_num = self._get_denoise_num()
        dllm_mask = dllm_mask.view(-1, block_size)
        is_masked = dllm_mask == DLLM_MASKED

        # get indices
        indices = is_masked.int().argmax(dim=1)
        ranges = torch.arange(0, denoise_num, device=indices.device, dtype=indices.dtype)
        indices = indices[:, None] + ranges[None, :]
        indices = indices % block_size

        dllm_unmasked = dllm_mask.clone()
        dllm_unmasked = dllm_unmasked.scatter(-1, indices, DLLM_UNMASKED)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()
    
    def reset_force_static_flags(self, dllm_mask: torch.Tensor):
        """Reset force_static_mask for completed blocks."""
        if self.force_static_mask is None:
            return
        
        block_size = self.dllm_config.block_length
        dllm_mask_view = dllm_mask.view(-1, block_size)
        
        # Clear flags for blocks that are fully unmasked or cached
        for block_idx in range(dllm_mask_view.shape[0]):
            block = dllm_mask_view[block_idx]
            # If no masked tokens remain, reset the flag
            if (block != DLLM_MASKED).all():
                self.force_static_mask[block_idx] = False

    @record_function('unmasking')
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """call."""
        strategy = self.dllm_config.unmasking_strategy
        if strategy is None:
            return dllm_mask

        # reshape to [num_blocks, block_size]
        block_size = self.dllm_config.block_length
        dllm_mask = dllm_mask.unflatten(0, (-1, block_size))

        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        dllm_mask[is_block_unmasked] = DLLM_CACHED

        dllm_mask = dllm_mask.flatten()
        token_ids = torch.where(dllm_mask != DLLM_MASKED, input_ids, token_ids)
        if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            dllm_mask = self.low_confidence_dynamic(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC_ENHANCED:
            dllm_mask = self.low_confidence_dynamic_enhanced(logits, token_ids, dllm_mask)
            # Clean up flags for completed blocks
            self.reset_force_static_flags(dllm_mask)
        elif strategy == UnmaskingStrategy.SEQUENTIAL:
            dllm_mask = self.sequential(dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return dllm_mask, token_ids