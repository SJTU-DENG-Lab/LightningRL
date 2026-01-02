import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project
    
    
    

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        json_content = json.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(json_content, dict) and "metadata" in json_content:
        total_generation_time = json_content["metadata"]["total_generation_time"]
        data = json_content["data"]
    else:
        # Old format compatibility
        total_generation_time = None
        data = json_content

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    num_forwards_list = []

    for i in range(len(data)):

        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]

        # Collect forward information
        if "num_forwards" in data[i]:
            num_forwards_list = num_forwards_list + data[i]["num_forwards"]

        if config.dataset.data_type == "math":
            data[i]["correctness"] = []
            # Extract final answer from GSM8K format (after ####)
            answer_str = data[i]["ground_truth_answer"]
            if "####" in answer_str:
                ground_truth = answer_str.split("####")[-1].strip()
            else:
                ground_truth = answer_str
            ground_truth_list = ground_truth_list + [ground_truth] * len(data[i]["extracted_output"])

    

    if config.dataset.data_type == "math":

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


    if config.dataset.data_type == "math":
        acc = sum(correctness_list)/len(correctness_list)
    else:
        num_task   = 0
        num_correct_task = 0
        for x in data:
            for y in x["correctness"]:
                num_correct_task += all(y)
                num_task += 1
        acc = num_correct_task / num_task if num_task else 0

    if config.rollout.output_unmasking_history == False:
        for i in range(len(data)):
            data[i]["step_map"] = []
    
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
        
        
        avg_len = sum(response_length_list)/len(response_length_list)

        # Calculate TPS and TPF metrics
        if total_generation_time is not None and len(num_forwards_list) > 0:
            total_tokens = sum(response_length_list)
            total_forwards = sum(num_forwards_list)

            tps = total_tokens / total_generation_time if total_generation_time > 0 else 0.0
            tpf = total_tokens / total_forwards if total_forwards > 0 else 0.0

            avg_forwards = total_forwards / len(num_forwards_list)

            save_and_print(f"acc: {acc}\navg length: {avg_len}")
            save_and_print(f"\n{'='*50}")
            save_and_print(f"Performance Metrics:")
            save_and_print(f"  Total Tokens: {total_tokens}")
            save_and_print(f"  Total Generation Time: {total_generation_time:.2f}s")
            save_and_print(f"  Total Forwards: {total_forwards}")
            save_and_print(f"  TPS (Tokens/Second): {tps:.2f}")
            save_and_print(f"  TPF (Tokens/Forward): {tpf:.2f}")
            save_and_print(f"  Avg Response Length: {avg_len:.2f}")
            save_and_print(f"  Avg Forwards per Sample: {avg_forwards:.2f}")
            save_and_print(f"{'='*50}")

            # Save performance stats to JSON
            import time as time_module
            stats_file = f"../{project_name}/results/performance_stats-{int(time_module.time())}.json"
            performance_stats = {
                "timestamp": time_module.time(),
                "dataset": dataset,
                "model": pretrained_model,
                "total_samples": len(data),
                "total_responses": len(response_length_list),
                "total_tokens": int(total_tokens),
                "total_generation_time_seconds": float(total_generation_time),
                "total_forwards": int(total_forwards),
                "tps": float(tps),
                "tpf": float(tpf),
                "mean_response_length": float(avg_len),
                "mean_forwards_per_sample": float(avg_forwards),
                "accuracy": float(acc)
            }
            with open(stats_file, "w") as stats_f:
                json.dump(performance_stats, stats_f, indent=2)
            save_and_print(f"\nPerformance stats saved to: {stats_file}")
        else:
            save_and_print(f"acc: {acc}\navg length: {avg_len}")
            save_and_print(f"\nWarning: total_generation_time or num_forwards data not available for TPS/TPF calculation")
