import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

from termcolor import cprint

# Disable all logging output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*is part of.*")

from omegaconf import OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


if __name__ == "__main__":
    config = get_config()

    # Disable transformers warnings
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Set cache directory: user-defined env vars take priority
    # Fall back to default path if not user-defined
    default_cache_base = (
        "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/hyz/tmp"
    )

    # TRITON_CACHE_DIR: respect user setting if present, otherwise use default
    if "TRITON_CACHE_DIR" not in os.environ:
        os.environ["TRITON_CACHE_DIR"] = f"{default_cache_base}/.triton_cache"

    # TORCHINDUCTOR_CACHE_DIR: respect user setting if present, otherwise use default
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{default_cache_base}/.torch_compile_cache"

    for cache_dir in [os.environ["TRITON_CACHE_DIR"], os.environ["TORCHINDUCTOR_CACHE_DIR"]]:
        os.makedirs(cache_dir, exist_ok=True)

    # Define subprocess runner that automatically passes env vars
    def run_subprocess(cmd, **kwargs):
        """Run subprocess, automatically passing current env vars"""
        env = os.environ.copy()
        env["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers warnings
        return subprocess.run(cmd, env=env, **kwargs)

    # Record cache files before training; only delete newly created ones afterward (avoids interfering with concurrent jobs)
    import atexit

    cache_dirs = [os.environ["TRITON_CACHE_DIR"], os.environ["TORCHINDUCTOR_CACHE_DIR"]]
    cache_files_before = set()
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            for root, _, files in os.walk(cache_dir):
                for f in files:
                    cache_files_before.add(os.path.join(root, f))

    def cleanup_new_cache():
        """Only delete cache files created during this training run; preserve files potentially in use by other jobs"""
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                continue
            for root, dirs, files in os.walk(cache_dir, topdown=False):
                for f in files:
                    path = os.path.join(root, f)
                    if path not in cache_files_before:
                        try:
                            os.remove(path)
                        except:
                            pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except:
                        pass

    atexit.register(cleanup_new_cache)

    # Set GPU devices from config
    if hasattr(config.experiment, "gpu_ids") and config.experiment.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.experiment.gpu_ids)
        print(f"Setting CUDA_VISIBLE_DEVICES={config.experiment.gpu_ids}")

    start_from_scratch = config.experiment.start_from_scratch
    project_name = config.experiment.project
    model_base = config.model.model_base

    from omegaconf import MISSING

    # Read use_value_model flag from config
    use_value_model_config = OmegaConf.select(config, "training.use_value_model", default=True)

    if OmegaConf.select(config, "model.value_base_model", default=MISSING) is not MISSING and use_value_model_config:
        have_value_model = True
    else:
        have_value_model = False

    if not use_value_model_config:
        cprint("[RL Loop] Value model disabled (use_value_model=False)", "yellow")
        cprint("[RL Loop] Will use Direct Reward strategy", "yellow")
    else:
        cprint("[RL Loop] Value model enabled (use_value_model=True)", "green")
        cprint("[RL Loop] Will use GAE strategy", "green")

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")

    def init_value_model(i, model_base, cfg):
        project_name = cfg.experiment.project
        script_name = "init_value_model.py"
        subprocess.run(
            f"python {script_name} config=../configs/{project_name}.yaml experiment.current_epoch={i} ",
            shell=True,
            cwd="train",
            check=True,
        )

    if start_from_scratch:
        os.makedirs(f"{project_name}/results", exist_ok=True)
        optimized_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        begin_with(
            f"{project_name}/results/results-rl-"
            + optimized_model.replace("/", ".")
            + "-"
            + config.dataset.train_dataset
            + ".txt"
        )
        begin_with(
            f"{project_name}/results/results-eval-"
            + optimized_model.replace("/", ".")
            + "-"
            + config.dataset.train_dataset
            + ".txt"
        )
        # Reset cursor to 0 when starting from scratch
        cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
        cursor_file.parent.mkdir(parents=True, exist_ok=True)
        cursor_file.write_text("0")
        cprint("[Start from Scratch] Reset data cursor to 0", "cyan")
        if have_value_model:
            init_value_model(1, model_base, config)
            optimized_value_model = "../" + project_name + "/ckpt/" + config.model.optimized_value_name
            begin_with(
                f"{project_name}/results/results-rl-"
                + optimized_value_model.replace("/", ".")
                + "-"
                + config.dataset.train_dataset
                + ".txt"
            )

    def sample(i, type, block_size=None, top_k=None, remasking_strategy=None):
        script_name = "rl_rollout.py"
        subprocess.run(
            f"python {script_name} "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd="sample",
            check=True,
        )

    def reward(i, type, is_code_task, block_size=None, top_k=None, remasking_strategy=None):
        if is_code_task:
            script_name = "rl_code_reward.py"
        else:
            script_name = "rl_reward.py"
        subprocess.run(
            f"python {script_name} "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd="reward",
            check=True,
        )

    def execute(i, type):
        subprocess.run(
            f"python rl_execute.py "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd="reward",
            check=True,
        )

    def train(i, target):
        if target == "policy":
            script_name = "train_policy.py"
        elif target == "value":
            script_name = "train_value.py"
        elif target == "policy_no_value":  # No value model: run Python script directly
            script_name = "train_policy_no_value.py"
            # policy_no_value does not require distributed execution; run as a single process
            subprocess.run(
                f"python train/{script_name} config=configs/{project_name}.yaml experiment.current_epoch={i} ",
                shell=True,
                check=True,
                env=os.environ.copy(),
            )
            return  # Early return

        run_subprocess(  # All other targets use accelerate distributed execution
            f"accelerate launch "
            f"--num_machines 1 "
            f"--machine_rank 0 "
            f"--main_process_ip 127.0.0.1 "
            f"--main_process_port 8888 "
            f"--config_file accelerate_configs/{config.experiment.deepspeed_file} "
            f"train/{script_name} "
            f"config=configs/{project_name}.yaml "
            f"experiment.current_epoch={i} ",
            shell=True,
            check=True,
        )

    if config.dataset.data_type == "code":
        is_code_task = True
    else:
        is_code_task = False

    # Value Model Pretraining Phase
    if start_from_scratch and have_value_model:
        from pathlib import Path

        value_pretrain_steps = OmegaConf.select(config, "training.value_pretrain_steps", default=0)
        value_pretrain_full_passes = OmegaConf.select(config, "training.value_pretrain_full_passes", default=0)
        value_pretrain_enabled = OmegaConf.select(config, "training.value_pretrain_enabled", default=False)

        # Method C: Auto-stop pretraining based on convergence metrics
        if value_pretrain_enabled:
            cprint("[Value Pretrain] Auto-stop pretraining enabled", "cyan")

            # Read threshold config
            ev_threshold = OmegaConf.select(config, "training.value_pretrain_ev_threshold", default=0.5)
            bias_threshold = OmegaConf.select(config, "training.value_pretrain_bias_threshold", default=0.1)
            corr_threshold = OmegaConf.select(config, "training.value_pretrain_corr_threshold", default=0.7)

            cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
            cursor_file.parent.mkdir(parents=True, exist_ok=True)
            cursor_file.write_text("0")
            cprint("[Value Pretrain] Reset cursor to 0", "cyan")

            # Create pretraining flag
            is_pretraining_file = Path(project_name) / "temp_data" / "is_pretraining.txt"
            is_pretraining_file.parent.mkdir(parents=True, exist_ok=True)
            is_pretraining_file.write_text(str(1))

            pretrain_epoch = 0
            converged = False

            while not converged:
                pretrain_epoch += 1
                cprint(f"[Value Pretrain] Epoch {pretrain_epoch}", "cyan")

                sample(0, "train")
                if is_code_task:
                    execute(0, "train")
                reward(0, "train", is_code_task)

                # Set pretraining flag in config
                config.training.is_value_pretraining = True
                train(0, target="value")
                config.training.is_value_pretraining = False

                # Read metrics and check convergence
                metrics_file = Path(project_name) / "temp_data" / "value_pretrain_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    ev = metrics.get("explained_variance", 0)
                    bias = metrics.get("bias", 999)
                    corr = metrics.get("correlation", 0)

                    ev_ok = ev > ev_threshold
                    bias_ok = abs(bias) < bias_threshold
                    corr_ok = corr > corr_threshold

                    status = f"EV: {ev:.4f}({'✓' if ev_ok else '✗'}) | Bias: {bias:.4f}({'✓' if bias_ok else '✗'}) | Corr: {corr:.4f}({'✓' if corr_ok else '✗'})"
                    cprint(f"[Value Pretrain] {status}", "cyan")

                    converged = ev_ok and bias_ok and corr_ok

                    if converged:
                        cprint("[Value Pretrain] Converged! Stopping pretraining.", "green")
                        break

            # Cleanup and reset
            cursor_file.write_text("0")
            if is_pretraining_file.exists():
                is_pretraining_file.unlink()
            cprint("[Value Pretrain] Reset cursor to 0 for main training", "green")

        # Method A: Full passes-based pretraining (prioritized)
        elif value_pretrain_full_passes > 0:
            cprint(
                f"[Value Pretrain] Starting value model pretraining for {value_pretrain_full_passes} full dataset passes...",
                "cyan",
            )

            completed_pass_file = Path(project_name) / "temp_data" / "completed_pass.txt"
            cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
            is_pretraining_file = Path(project_name) / "temp_data" / "is_pretraining.txt"

            # Reset cursor to 0 before pretraining
            cursor_file.parent.mkdir(parents=True, exist_ok=True)
            cursor_file.write_text("0")
            cprint("[Value Pretrain] Reset cursor to 0 before pretraining.", "cyan")

            # Create pretraining flag file (to disable reshuffling during pretrain)
            is_pretraining_file.parent.mkdir(parents=True, exist_ok=True)
            is_pretraining_file.write_text(str(1))
            cprint("[Value Pretrain] Created pretraining flag (reshuffling disabled)", "cyan")

            # Clear previous completed_pass flag
            if completed_pass_file.exists():
                completed_pass_file.unlink()

            pretrain_epoch = 0
            completed_pass_count = 0

            while completed_pass_count < value_pretrain_full_passes:
                pretrain_epoch += 1
                cprint(
                    f"[Value Pretrain] Epoch {pretrain_epoch} (completed passes: {completed_pass_count}/{value_pretrain_full_passes})",
                    "cyan",
                )

                sample(0, "train")
                if is_code_task:
                    execute(0, "train")
                reward(0, "train", is_code_task)
                cprint(f"[Value Pretrain] Epoch {pretrain_epoch} - Training...", "cyan")
                train(0, target="value")

                # Check if completed a full pass
                if completed_pass_file.exists():
                    completed_pass_count += 1
                    cprint(
                        f"[Value Pretrain] Completed pass {completed_pass_count}/{value_pretrain_full_passes}", "green"
                    )
                    completed_pass_file.unlink()

            cprint(
                f"[Value Pretrain] Pretraining completed. Total epochs: {pretrain_epoch}, Total passes: {completed_pass_count}",
                "green",
            )

            # Reset cursor to 0 and clear flags for main training
            cursor_file.write_text("0")
            if completed_pass_file.exists():
                completed_pass_file.unlink()
            if is_pretraining_file.exists():
                is_pretraining_file.unlink()
                cprint("[Value Pretrain] Removed pretraining flag (reshuffling enabled for main training)", "cyan")
            cprint("[Value Pretrain] Reset cursor to 0 for main training.", "green")

        # Method B: Steps-based pretraining (original, mutually exclusive)
        elif value_pretrain_steps > 0:
            cprint(
                f"[Value Pretrain] Starting value model pretraining until {value_pretrain_steps} global steps...",
                "cyan",
            )

            # Track cumulative global steps across epochs
            cumulative_global_steps = 0
            pretrain_epoch = 0

            while cumulative_global_steps < value_pretrain_steps:
                pretrain_epoch += 1
                cprint(
                    f"[Value Pretrain] Epoch {pretrain_epoch} (cumulative steps: {cumulative_global_steps}/{value_pretrain_steps}) - Rolling out...",
                    "cyan",
                )
                sample(0, "train")
                if is_code_task:
                    execute(0, "train")
                reward(0, "train", is_code_task)
                cprint(f"[Value Pretrain] Epoch {pretrain_epoch} - Training...", "cyan")
                train(0, target="value")

                # Read global steps from this epoch
                steps_file = Path(project_name) / "temp_data" / "value_train_steps.txt"
                if steps_file.exists():
                    try:
                        epoch_steps = int(steps_file.read_text().strip())
                        cumulative_global_steps += epoch_steps
                        cprint(
                            f"[Value Pretrain] Epoch {pretrain_epoch} completed: +{epoch_steps} steps, total: {cumulative_global_steps}/{value_pretrain_steps}",
                            "cyan",
                        )
                    except (ValueError, OSError):
                        cprint("[Value Pretrain] Warning: Could not read steps file", "yellow")

            cprint(
                f"[Value Pretrain] Pretraining completed. Total epochs: {pretrain_epoch}, Total steps: {cumulative_global_steps}",
                "green",
            )

            # Reset cursor to initial position for main training
            cursor_file = Path(project_name) / "temp_data" / "data_cursor.txt"
            cursor_file.parent.mkdir(parents=True, exist_ok=True)
            cursor_file.write_text("0")
            cprint("[Value Pretrain] Reset data cursor to 0 for main training.", "green")

    i = config.experiment.current_epoch

    while i <= config.experiment.total_step:
        sample(i, "train")
        if is_code_task:
            execute(i, "train")
        reward(i, "train", is_code_task)

        # Data filter resample logic: if prompts were filtered, resample to reach target count
        data_filter_enabled = OmegaConf.select(config, "rollout.data_filter", default=False)
        resample_enabled = OmegaConf.select(config, "rollout.data_filter_resample_enabled", default=True)
        if data_filter_enabled and resample_enabled:
            import json
            from pathlib import Path

            stats_file = Path(project_name) / "temp_data" / "data_filter_stats.txt"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                filtered_count = stats.get("filtered_count", 0)
                kept_count = stats.get("kept_count", 0)
                target_count = stats.get("target_count", 0)
                if kept_count < target_count and filtered_count > 0:
                    cprint(
                        f"[RL Loop] Resampling: filtered={filtered_count}, kept={kept_count}, target={target_count}",
                        "cyan",
                    )
                    sample(i, "train")
                    if is_code_task:
                        execute(i, "train")
                    reward(i, "train", is_code_task)

        if have_value_model:
            train(i, target="value")
            train(i, target="policy")
        else:
            # Without value model: compute advantage first, then train policy
            train(i, target="policy_no_value")  # Compute advantage
            train(i, target="policy")  # Train policy

        if i % config.experiment.eval_every == 0:
            remasking_strategy_list = config.evaluation.remasking_strategy
            top_k_list = config.evaluation.top_k
            block_size = config.evaluation.block_size
            for j in range(len(remasking_strategy_list)):
                remasking_strategy = remasking_strategy_list[j]
                top_k = top_k_list[j]
                sample(i, "evaluation", block_size=block_size, top_k=top_k, remasking_strategy=remasking_strategy)
                if is_code_task:
                    execute(i, "evaluation")
                reward(
                    i,
                    "evaluation",
                    is_code_task,
                    block_size=block_size,
                    top_k=top_k,
                    remasking_strategy=remasking_strategy,
                )

        i += 1
