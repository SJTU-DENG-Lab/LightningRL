"""
Value Model evaluation metrics.

This module provides functions to compute and format metrics for evaluating
the performance of Value Model training.
"""

import torch
from typing import Dict, Optional, Tuple


def compute_value_metrics(
    returns: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    correctness: Optional[torch.Tensor] = None,
    seq_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics for evaluating Value Model performance.
    
    Args:
        returns: tensor of shape (N, L) or (N,) - actual returns R
        values: tensor of shape (N, L) or (N,) - predicted values V
        mask: optional tensor of shape (N, L) - valid positions mask (p_mask)
        correctness: optional tensor of shape (B,) - bool correctness labels per sequence
        seq_ids: optional tensor of shape (N,) - mapping from row to original sequence id
    
    Returns:
        dict with the following metrics:
            - explained_variance: 1 - Var(R-V) / Var(R), in [-inf, 1]
            - bias: mean(R - V), should be close to 0
            - mse: mean((R - V)^2)
            - mean_return: mean(R)
            - std_return: std(R)
            - mean_value: mean(V)
            - std_value: std(V)
            - mean_advantage: mean(A), where A = R - V
            - std_advantage: std(A)
            - correlation: Pearson correlation between R and V
            - value_gap: mean(V|correct) - mean(V|incorrect), if correctness provided
    """
    # Flatten if needed and apply mask
    if mask is not None:
        # Extract only valid positions
        r = returns[mask].float()
        v = values[mask].float()
    else:
        r = returns.flatten().float()
        v = values.flatten().float()
    
    if r.numel() == 0:
        return {"error": "No valid data points"}
    
    # Advantage
    a = r - v
    
    # Basic statistics
    mean_r = r.mean().item()
    std_r = r.std().item() if r.numel() > 1 else 0.0
    mean_v = v.mean().item()
    std_v = v.std().item() if v.numel() > 1 else 0.0
    mean_a = a.mean().item()
    std_a = a.std().item() if a.numel() > 1 else 0.0
    
    # MSE
    mse = (a ** 2).mean().item()
    
    # Explained Variance: 1 - Var(R-V) / Var(R)
    var_r = r.var().item() if r.numel() > 1 else 0.0
    var_a = a.var().item() if a.numel() > 1 else 0.0
    if var_r > 1e-8:
        explained_variance = 1.0 - var_a / var_r
    else:
        explained_variance = 0.0  # Undefined when Var(R) = 0
    
    # Bias
    bias = mean_a
    
    # Pearson Correlation
    if std_r > 1e-8 and std_v > 1e-8:
        # corr = cov(R, V) / (std(R) * std(V))
        cov_rv = ((r - mean_r) * (v - mean_v)).mean().item()
        correlation = cov_rv / (std_r * std_v)
    else:
        correlation = 0.0
    
    metrics = {
        "explained_variance": explained_variance,
        "bias": bias,
        "mse": mse,
        "mean_return": mean_r,
        "std_return": std_r,
        "mean_value": mean_v,
        "std_value": std_v,
        "mean_advantage": mean_a,
        "std_advantage": std_a,
        "correlation": correlation,
    }
    
    # Value gap between correct and incorrect samples
    if correctness is not None and seq_ids is not None:
        try:
            # Map correctness to each row using seq_ids
            row_correctness = correctness[seq_ids.long()]
            
            if mask is not None:
                # We need to get correctness for each valid position
                # This is more complex - need to track which positions belong to which seq
                # For simplicity, compute at sequence level
                pass
            else:
                correct_mask = row_correctness.bool()
                if correct_mask.any() and (~correct_mask).any():
                    v_flat = values.flatten().float()
                    correct_mask_flat = correct_mask.unsqueeze(1).expand_as(values).flatten()
                    
                    v_correct = v_flat[correct_mask_flat].mean().item()
                    v_incorrect = v_flat[~correct_mask_flat].mean().item()
                    metrics["value_gap"] = v_correct - v_incorrect
                    metrics["mean_value_correct"] = v_correct
                    metrics["mean_value_incorrect"] = v_incorrect
        except Exception:
            pass  # Skip value_gap if computation fails
    
    return metrics


def compute_sequence_level_metrics(
    per_seq_reward: torch.Tensor,
    seq_values: torch.Tensor,
    correctness: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics at sequence level (one value per sequence).
    
    Args:
        per_seq_reward: tensor of shape (B,) - reward for each sequence
        seq_values: tensor of shape (B,) - mean predicted value for each sequence
        correctness: optional tensor of shape (B,) - bool correctness labels
    
    Returns:
        dict with sequence-level metrics
    """
    r = per_seq_reward.float()
    v = seq_values.float()
    a = r - v
    
    metrics = {
        "seq_mean_return": r.mean().item(),
        "seq_std_return": r.std().item() if r.numel() > 1 else 0.0,
        "seq_mean_value": v.mean().item(),
        "seq_std_value": v.std().item() if v.numel() > 1 else 0.0,
        "seq_mean_advantage": a.mean().item(),
        "seq_std_advantage": a.std().item() if a.numel() > 1 else 0.0,
    }
    
    # Explained variance at sequence level
    var_r = r.var().item() if r.numel() > 1 else 0.0
    var_a = a.var().item() if a.numel() > 1 else 0.0
    if var_r > 1e-8:
        metrics["seq_explained_variance"] = 1.0 - var_a / var_r
    else:
        metrics["seq_explained_variance"] = 0.0
    
    # Value gap
    if correctness is not None:
        correct_mask = correctness.bool()
        if correct_mask.any() and (~correct_mask).any():
            v_correct = v[correct_mask].mean().item()
            v_incorrect = v[~correct_mask].mean().item()
            metrics["seq_value_gap"] = v_correct - v_incorrect
            metrics["seq_mean_value_correct"] = v_correct
            metrics["seq_mean_value_incorrect"] = v_incorrect
    
    return metrics


def format_value_metrics(
    metrics: Dict[str, float],
    epoch: int = 0,
    prefix: str = "",
    output_file: Optional[str] = None,
    ev_threshold: float = 0.5,
    bias_threshold: float = 0.1,
    corr_threshold: float = 0.7
) -> str:
    """
    Format metrics into a readable string for logging.

    Args:
        metrics: dict of metric name -> value
        epoch: current epoch number
        prefix: optional prefix for the output
        output_file: optional file path to append metrics to
        ev_threshold: explained_variance threshold for "good" status
        bias_threshold: bias threshold (absolute) for "good" status
        corr_threshold: correlation threshold for "good" status

    Returns:
        Formatted string for logging
    """
    lines = []
    
    if prefix:
        lines.append(f"{prefix}")
    
    lines.append(f"[Value Model Metrics] epoch={epoch}")
    
    # Core metrics
    if "explained_variance" in metrics:
        ev = metrics["explained_variance"]
        ev_status = "✓" if ev > ev_threshold else ("△" if ev > 0 else "✗")
        lines.append(f"  explained_variance: {ev:.4f} {ev_status}")

    if "bias" in metrics:
        bias = metrics["bias"]
        bias_status = "✓" if abs(bias) < bias_threshold else "△"
        lines.append(f"  bias: {bias:.4f} {bias_status}")

    if "mse" in metrics:
        lines.append(f"  mse: {metrics['mse']:.4f}")

    if "correlation" in metrics:
        corr = metrics["correlation"]
        corr_status = "✓" if corr > corr_threshold else ("△" if corr > 0.3 else "✗")
        lines.append(f"  correlation: {corr:.4f} {corr_status}")
    
    # Return/Value statistics
    if "mean_return" in metrics and "mean_value" in metrics:
        lines.append(f"  return: mean={metrics['mean_return']:.4f}, std={metrics.get('std_return', 0):.4f}")
        lines.append(f"  value:  mean={metrics['mean_value']:.4f}, std={metrics.get('std_value', 0):.4f}")
    
    # Advantage statistics
    if "mean_advantage" in metrics:
        lines.append(f"  advantage: mean={metrics['mean_advantage']:.4f}, std={metrics.get('std_advantage', 0):.4f}")
    
    # Value gap
    if "value_gap" in metrics:
        gap = metrics["value_gap"]
        gap_status = "✓" if gap > 0 else "✗"
        lines.append(f"  value_gap (correct-incorrect): {gap:.4f} {gap_status}")
    
    if "seq_value_gap" in metrics:
        gap = metrics["seq_value_gap"]
        gap_status = "✓" if gap > 0 else "✗"
        lines.append(f"  seq_value_gap: {gap:.4f} {gap_status}")
    
    result = "\n".join(lines)
    
    # Write to file if output_file is provided
    if output_file:
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(result + "\n")
        except Exception:
            pass  # Silently fail if file write fails
    
    return result


def collect_value_metrics_from_dataset(dataset, accelerator=None) -> Dict[str, float]:
    """
    Collect and compute value metrics from a TrainDataset after GAE computation.
    
    Args:
        dataset: TrainDataset object with Return, old_values, p_mask, etc.
        accelerator: optional accelerator for distributed gathering
    
    Returns:
        dict with computed metrics
    """
    # Get data from dataset
    returns = dataset.Return  # (N, L)
    values = dataset.old_values  # (N, L)
    mask = dataset.p_mask  # (N, L)
    
    # Compute token-level metrics
    metrics = compute_value_metrics(returns, values, mask)
    
    # Try to compute sequence-level metrics if possible
    if hasattr(dataset, 'per_seq_reward') and hasattr(dataset, 'seq_ids'):
        try:
            # Aggregate values per sequence
            B = dataset.per_seq_reward.numel()
            seq_values = torch.zeros(B)
            seq_counts = torch.zeros(B)
            
            for row in range(dataset.p_mask.size(0)):
                s = int(dataset.seq_ids[row].item())
                pm = dataset.p_mask[row]
                if pm.any():
                    v_row = dataset.old_values[row][pm].mean()
                    seq_values[s] += v_row
                    seq_counts[s] += 1
            
            # Average
            valid_seqs = seq_counts > 0
            seq_values[valid_seqs] /= seq_counts[valid_seqs]
            
            # Compute sequence-level metrics
            seq_metrics = compute_sequence_level_metrics(
                dataset.per_seq_reward[valid_seqs],
                seq_values[valid_seqs],
            )
            metrics.update(seq_metrics)
        except Exception:
            pass  # Skip if computation fails

    return metrics


def check_pretrain_convergence(
    metrics: Dict[str, float],
    ev_threshold: float,
    bias_threshold: float,
    corr_threshold: float
) -> Tuple[bool, Dict[str, bool], str]:
    """
    Check whether pretraining has converged

    Args:
        metrics: dict of metric name -> value
        ev_threshold: explained_variance threshold
        bias_threshold: bias threshold (absolute)
        corr_threshold: correlation threshold

    Returns:
        (converged, per-metric status dict, status string)
    """
    ev = metrics.get("explained_variance", 0)
    bias = metrics.get("bias", 999)
    corr = metrics.get("correlation", 0)

    ev_ok = ev > ev_threshold
    bias_ok = abs(bias) < bias_threshold
    corr_ok = corr > corr_threshold

    status = {
        "explained_variance": ev_ok,
        "bias": bias_ok,
        "correlation": corr_ok
    }

    status_str = (
        f"EV: {ev:.4f}({'✓' if ev_ok else '✗'}) | "
        f"Bias: {bias:.4f}({'✓' if bias_ok else '✗'}) | "
        f"Corr: {corr:.4f}({'✓' if corr_ok else '✗'})"
    )

    converged = ev_ok and bias_ok and corr_ok
    return converged, status, status_str
