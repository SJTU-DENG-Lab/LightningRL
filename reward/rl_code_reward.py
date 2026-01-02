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
    
    if config.experiment.current_epoch == 1:
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



    response_length_list = []
    num_task   = 0
    num_correct_task = 0
    final_data = []
    for i in range(len(data)):
        response_length_list = response_length_list + data[i]["response_length"]
        correctness_list = data[i]["correctness"]
        lengths = data[i]["response_length"]
        
        # Count fully correct samples
        for corr in correctness_list:
            num_correct_task += all(corr)
            num_task += 1
        
        # Apply length penalty: mark as failed if too long
        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness_list[j] = [False] * len(correctness_list[j])
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness_list[j] = [False] * len(correctness_list[j])
        
        # Calculate TPF (Tokens Per Forward) for each sample in this prompt
        tpf_list = []
        for j in range(len(lengths)):
            if "num_forwards" in data[i] and j < len(data[i]["num_forwards"]) and data[i]["num_forwards"][j] > 0:
                tpf = lengths[j] / data[i]["num_forwards"][j]
            else:
                tpf = 0.0
            tpf_list.append(tpf)
        
        # Normalize TPF to [0, 1] for this prompt's samples
        if len(tpf_list) > 0 and max(tpf_list) > 0:
            min_tpf = min(tpf_list)
            max_tpf = max(tpf_list)
            if max_tpf > min_tpf:
                normalized_tpf = [(tpf - min_tpf) / (max_tpf - min_tpf) for tpf in tpf_list]
            else:
                normalized_tpf = [0.5] * len(tpf_list)  # All same, give middle value
        else:
            normalized_tpf = [0.0] * len(tpf_list)
            min_tpf, max_tpf = 0, 0
        
        # Calculate rewards using pass_rate * TPF strategy
        # If pass_rate > 0: reward = pass_rate * normalized_tpf (multiplication)
        # If pass_rate == 0: reward = -1 + normalized_tpf * 0.5 (addition, same as math tasks)
        rewards = []
        for j in range(len(correctness_list)):
            if len(correctness_list[j]) > 0:
                pass_rate = sum(correctness_list[j]) / len(correctness_list[j])
            else:
                pass_rate = 0
            
            if pass_rate > 0:  # Has passed some tests
                reward = pass_rate * normalized_tpf[j]  # Multiplication: [0, 1]
            else:  # Complete failure
                reward = -1.0 + normalized_tpf[j] * 0.5  # Addition: [-1, -0.5]
            
            rewards.append(reward)
        
        data[i]["rewards"] = rewards
        
        if config.experiment.function == "train":
            # Filter out prompts where TPF variance is too small (no learning signal)
            # Same as math tasks: threshold = 0.1
            tpf_variance = max_tpf - min_tpf if len(tpf_list) > 0 else 0
            if tpf_variance < 0.1:
                continue

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                final_data.append(data_i)
        
        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []


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

        output_text = f"train step: {config.experiment.current_epoch}  "
        
        if config.experiment.function == "train":
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}"
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}"
        save_and_print(output_text)
