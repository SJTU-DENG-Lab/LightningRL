import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance


def compute_dense_rewards(step_map, response_tokens, correctness, tokenizer, config, 
                          ground_truth_answer=None):
    """
    Compute dense rewards for each step based on edit distance progress.
    
    For CORRECT samples:
        - reward = alpha_token * tokens_in_step + beta_edit * (prev_dist - curr_dist)
        - tokens_in_step: number of tokens decoded in this step
        - (prev_dist - curr_dist): progress towards final output this step
        - Encourages fast decoding AND meaningful progress each step
    
    For INCORRECT samples:
        - reward = beta_edit * (prev_dist - curr_dist)
        - No speed reward, only measures progress towards ground truth
    
    Args:
        step_map: list of step ids for each token position
        response_tokens: list of token ids for the response
        correctness: bool indicating if the final answer is correct
        tokenizer: tokenizer for decoding tokens
        config: configuration object with dense_reward settings
        ground_truth_answer: the full ground truth answer text (for incorrect samples)
    
    Returns:
        step_rewards: dict mapping step_id to reward value
        scalar_reward: average of all step rewards (for monitoring)
    """
    if not step_map or len(step_map) == 0:
        return {}, 0.0
    
    # Get dense reward config with defaults
    dense_cfg = OmegaConf.select(config, "dense_reward", default=None)
    if dense_cfg is None or not OmegaConf.select(dense_cfg, "enabled", default=True):
        # Fall back to sparse reward behavior
        return None, None
    
    alpha_token = OmegaConf.select(dense_cfg, "alpha_token", default=0.1)
    beta_edit = OmegaConf.select(dense_cfg, "beta_edit", default=1.0)
    
    # Get unique steps in order
    unique_steps = sorted(set(step_map))
    S = len(unique_steps)
    
    # Decode the final output text (for correct samples comparison)
    try:
        final_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    except Exception:
        final_text = ""
    
    # Determine target text for edit distance comparison
    if correctness:
        target_text = final_text
    else:
        target_text = ground_truth_answer if ground_truth_answer else ""
    
    step_rewards = {}
    prev_dist = len(target_text)  # Initial distance: from empty string to target
    
    for i, step_id in enumerate(unique_steps):
        # Count tokens decoded in this step
        tokens_in_step = sum(1 for s in step_map if s == step_id)
        
        # Get cumulative text up to and including this step
        mask = [s <= step_id for s in step_map]
        current_tokens = [t for t, m in zip(response_tokens, mask) if m]
        
        # Decode to text
        try:
            current_text = tokenizer.decode(current_tokens, skip_special_tokens=True)
        except Exception:
            current_text = ""
        
        # Calculate current distance to target
        curr_dist = levenshtein_distance(current_text, target_text)
        
        # Edit distance progress: how much closer we got this step
        edit_progress = prev_dist - curr_dist
        
        if correctness:
            # CORRECT sample: speed reward + edit progress reward
            r_t = alpha_token * tokens_in_step + beta_edit * edit_progress
        else:
            # INCORRECT sample: only edit progress reward (no speed bonus)
            r_t = beta_edit * edit_progress
        
        step_rewards[step_id] = r_t
        prev_dist = curr_dist  # Update for next iteration
    
    # Add +1 bonus to the last step for correct samples
    if correctness and unique_steps:
        last_step_id = unique_steps[-1]
        step_rewards[last_step_id] = step_rewards.get(last_step_id, 0.0) + 1.0
    
    # Calculate scalar reward (sum of all step rewards) for monitoring
    scalar_reward = sum(step_rewards.values()) if step_rewards else 0.0
    
    return step_rewards, scalar_reward


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
    
    # Load tokenizer for dense reward computation
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

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

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    ground_truth_full_list = []  # Full ground truth answer (with reasoning steps)
    response_length_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]

        # Extract ground truth from answer field (GSM8K format)
        answer_str = data[i]["ground_truth_answer"]
        # Store full answer for dense reward comparison
        ground_truth_full_list = ground_truth_full_list + [answer_str] * len(data[i]["extracted_output"])
        # Extract just the final answer for correctness checking
        if "####" in answer_str:
            ground_truth = answer_str.split("####")[-1].strip()
        else:
            ground_truth = answer_str
        ground_truth_list = ground_truth_list + [ground_truth] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    async def get_correctness():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = []
        for i in range(len(index_list)):
            tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
        results = await asyncio.gather(*tasks)
        return results

    correctness_list = asyncio.run(get_correctness())
    for i in range(len(index_list)):
        index_i = index_list[i]
        data[index_i]["correctness"].append(correctness_list[i])



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



    # Check if dense reward is enabled (only for training, not evaluation)
    is_training = config.experiment.function == "train"
    dense_reward_enabled = is_training and OmegaConf.select(config, "dense_reward.enabled", default=False)
    
    if is_training:
        if dense_reward_enabled:
            cprint("[INFO] Dense reward is ENABLED. Computing per-step rewards...", "cyan")
        else:
            cprint("[INFO] Dense reward is DISABLED. Using sparse rewards.", "cyan")

    final_data = []
    for i in tqdm(range(len(data)), desc="Processing prompts", unit="prompt"):
        correctness = data[i]["correctness"]
        lengths = data[i]["response_length"]

        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness[j] = False
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness[j] = False

        # Get full ground truth answer for this prompt (for incorrect samples comparison)
        ground_truth_answer_full = data[i]["ground_truth_answer"]
        
        # Compute dense rewards if enabled
        if dense_reward_enabled:
            dense_rewards_list = []
            step_rewards_list = []
            for j in range(len(correctness)):
                step_map_j = data[i]["step_map"][j] if j < len(data[i]["step_map"]) else []
                response_text = data[i]["full_output"][j] if j < len(data[i]["full_output"]) else ""
                
                # Tokenize the response to get token ids
                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                
                # Compute dense rewards (speed reward = alpha * tokens_in_step)
                step_rewards, scalar_reward = compute_dense_rewards(
                    step_map_j, 
                    response_tokens, 
                    correctness[j],
                    tokenizer,
                    config,
                    ground_truth_answer=ground_truth_answer_full
                )
                
                if step_rewards is not None:
                    step_rewards_list.append(step_rewards)
                    dense_rewards_list.append(scalar_reward)
                else:
                    # Fall back to sparse reward if dense reward disabled/failed
                    step_rewards_list.append({})
                    if correctness[j]:
                        # Sparse fallback: correct=1
                        dense_rewards_list.append(1.0)
                    else:
                        # Sparse fallback: incorrect=0
                        dense_rewards_list.append(-1)
            
            rewards = dense_rewards_list
            data[i]["step_rewards_list"] = step_rewards_list
        else:
            # Sparse reward logic: correct=1, incorrect=0
            rewards = []
            for j in range(len(correctness)):
                if correctness[j]:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

        data[i]["rewards"] = rewards
        
        if config.experiment.function == "train":
            
            # Data filter: keep only samples with 0 < mean_correctness < 1
            # (has learning signal and not all correct)
            data_filter = OmegaConf.select(config, "rollout.data_filter", default=False)
            mean_correctness = sum(correctness) / len(correctness) if correctness else 0
            
            if data_filter:
                # Data filter: 0 < acc < 1 and at least one correct
                if not (0 < mean_correctness < 1.0 and max(correctness) == 1):
                    continue
            else:
                # Original logic: only filter all-incorrect
                if not any(correctness):
                    continue

            for j in range(len(rewards)):
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["response"] = data[i]["full_output"][j]
                data_i["step_map"] = data[i]["step_map"][j]
                
                # Add step_rewards if dense reward is enabled
                if dense_reward_enabled and "step_rewards_list" in data[i]:
                    # Convert int keys to string for JSON serialization
                    step_rewards_dict = data[i]["step_rewards_list"][j]
                    data_i["step_rewards"] = {str(k): v for k, v in step_rewards_dict.items()}
                
                # Add correctness for NLL loss (VAPO)
                data_i["correctness"] = correctness[j]
                
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
        
        acc = sum(correctness_list)/len(correctness_list)
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
