import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project

    if config.experiment.current_epoch <= 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
    

    if config.experiment.function == "train":
        shrink = config.training.shrink
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    
    

    
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        json_content = json.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(json_content, dict) and "metadata" in json_content:
        data = json_content["data"]
    else:
        # Old format compatibility
        data = json_content



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]






    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst



    # ===== Reward Configuration =====
    tpf_coefficient = OmegaConf.select(config, "reward.tpf_coefficient", default=0.1)
    filter_tpf_threshold = OmegaConf.select(config, "reward.filter_tpf_threshold", default=0.1)
    filter_all_wrong = OmegaConf.select(config, "reward.filter_all_wrong", default=True)
    # TPF normalization mode: supports 3 modes (none/z_score/minmax)
    # Backward compatibility: map legacy tpf_norm boolean to tpf_norm_mode
    tpf_norm_legacy = OmegaConf.select(config, "reward.tpf_norm", default=None)
    if tpf_norm_legacy is not None:
        # Legacy config found: true="z_score", false="none"
        tpf_norm_mode = "z_score" if tpf_norm_legacy else "none"
    else:
        # Use new tpf_norm_mode config
        tpf_norm_mode = OmegaConf.select(config, "reward.tpf_norm_mode", default="minmax")
    acc_norm = OmegaConf.select(config, "reward.acc_norm", default=False)

    response_length_list = []
    tpf_list = []  # Collect TPF statistics across all samples
    num_task   = 0
    num_correct_task = 0
    final_data = []
    for i in range(len(data)):
        response_length_list = response_length_list + data[i]["response_length"]
        lengths = data[i]["response_length"]

        # Calculate TPF (Tokens Per Forward) for each sample in this prompt
        local_tpf_list = []
        for j in range(len(lengths)):
            if "num_forwards" in data[i] and j < len(data[i]["num_forwards"]) and data[i]["num_forwards"][j] > 0:
                tpf = lengths[j] / data[i]["num_forwards"][j]
            else:
                tpf = 0.0
            local_tpf_list.append(tpf)
            tpf_list.append(tpf)  # Collect for statistics

        # Calculate base rewards from correctness (using pass rate)
        if acc_norm:
            # Z-score normalize on pass rates
            pass_rates = []
            for x in data[i]["correctness"]:
                pass_rate = sum(x) / len(x) if len(x) > 0 else 0.0
                pass_rates.append(pass_rate)
                num_correct_task += all(x)
                num_task += 1
            base_rewards = z_score_normalize(pass_rates)
        else:
            # Direct pass rate (0~1)
            base_rewards = []
            for x in data[i]["correctness"]:
                pass_rate = sum(x) / len(x) if len(x) > 0 else 0.0
                base_rewards.append(pass_rate)
                num_correct_task += all(x)
                num_task += 1

        # Prompt-level TPF normalization
        if tpf_norm_mode == "none":
            # No normalization: use raw TPF * tpf_coefficient
            speed_reward_list = [tpf_coefficient * tpf for tpf in local_tpf_list]
        elif tpf_norm_mode == "z_score":
            # Z-score normalization
            mean_tpf = sum(local_tpf_list) / len(local_tpf_list)
            std_tpf = (sum((x - mean_tpf) ** 2 for x in local_tpf_list) / len(local_tpf_list)) ** 0.5
            if std_tpf > 0:
                speed_reward_list = [tpf_coefficient * (tpf - mean_tpf) / std_tpf for tpf in local_tpf_list]
            else:
                speed_reward_list = [0.0] * len(local_tpf_list)
        elif tpf_norm_mode == "minmax":
            # Min-max normalization to [0, 1]
            tpf_min = min(local_tpf_list)
            tpf_max = max(local_tpf_list)
            tpf_range = tpf_max - tpf_min
            if tpf_range > 0:
                speed_reward_list = [(tpf - tpf_min) / tpf_range for tpf in local_tpf_list]
            else:
                speed_reward_list = [0.5] * len(local_tpf_list)
        else:
            raise ValueError(f"Invalid tpf_norm_mode: {tpf_norm_mode}. Must be 'none', 'z_score', or 'minmax'")

        # Calculate new reward: base_reward + speed_reward
        rewards = []
        for j in range(len(base_rewards)):
            rewards.append(base_rewards[j] + speed_reward_list[j])
        data[i]["ground_truth_answer"] = rewards

        if config.experiment.function == "train":
            # Filter 1: All-wrong filter (configurable)
            if filter_all_wrong:
                all_wrong = all(sum(x) == 0 for x in data[i]["correctness"])
                if all_wrong:
                    continue

            # Filter 2: TPF range filter (using original TPF)
            if max(local_tpf_list) - min(local_tpf_list) < filter_tpf_threshold:
                continue

            # Filter 3: All-wrong check (check correctness not reward)
            if all(sum(x) == 0 for x in data[i]["correctness"]):
                continue

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                # Add truncated field (default to False if not present for backward compatibility)
                data_i["truncated"] = data[i].get("truncated", [False] * len(rewards))[j]
                # Pass through sample_idx and resp_idx for correct advantage grouping
                data_i["sample_idx"] = data[i].get("sample_idx", [i] * len(rewards))[j]
                data_i["resp_idx"] = data[i].get("resp_idx", list(range(len(rewards))))[j]
                final_data.append(data_i)

        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []


    # Calculate Efficient Prompt Ratio (proportion of prompts where tpf max-min < 0.01)
    efficient_prompt_count = 0
    total_prompt_count = 0

    for i in range(len(data)):
        if "num_forwards" in data[i] and len(data[i]["num_forwards"]) > 0:
            local_tpf_list = []
            lengths = data[i]["response_length"]
            for j in range(len(lengths)):
                if data[i]["num_forwards"][j] > 0:
                    tpf = lengths[j] / data[i]["num_forwards"][j]
                else:
                    tpf = 0.0
                local_tpf_list.append(tpf)

            if len(local_tpf_list) > 0:
                tpf_range = max(local_tpf_list) - min(local_tpf_list)
                if tpf_range < 0.01:
                    efficient_prompt_count += 1
                total_prompt_count += 1

    efficient_prompt_ratio = efficient_prompt_count / total_prompt_count if total_prompt_count > 0 else 0.0


    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)


    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = num_correct_task / num_task if num_task else 0
        avg_len = sum(response_length_list)/len(response_length_list)

        # Calculate TPF statistics (only for non-zero TPF values)
        valid_tpf = [t for t in tpf_list if t > 0]
        if valid_tpf:
            avg_tpf = sum(valid_tpf) / len(valid_tpf)
            min_tpf = min(valid_tpf)
            max_tpf = max(valid_tpf)
            tpf_stats = f"avg_tpf: {avg_tpf:.2f}  min_tpf: {min_tpf:.2f}  max_tpf: {max_tpf:.2f}"
        else:
            tpf_stats = "avg_tpf: N/A"

        output_text = f"train step: {config.experiment.current_epoch}  "

        if config.experiment.function == "train":
            output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}  {tpf_stats}  efficient_prompt_ratio: {efficient_prompt_ratio:.4f}"
        else:
            output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}  {tpf_stats}"
        save_and_print(output_text)
