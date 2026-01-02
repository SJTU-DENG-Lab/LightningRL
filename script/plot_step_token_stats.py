# ============== 配置区（手动修改此处） ==============
INPUT_JSON = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/hyz/Fast-RL/sdar_eval/temp_data/hyz.Fast-RL.rl_sdar_with_value_4B_Math.ckpt.epoch-80-policy-GSM8K.json"
OUTPUT_PNG = None  # None则自动生成到同目录，或指定如 "sdar_eval/results/step_stats.png"
# ==================================================

import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# 获取脚本所在目录的上级目录（Fast-RL根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_data(json_path):
    """读取JSON文件，兼容两种格式"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 兼容带metadata和不带metadata的格式
    if isinstance(data, dict) and "data" in data:
        samples = data["data"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("无法识别的JSON格式")
    
    return samples


def compute_step_stats(samples):
    """
    从samples中提取所有response的step统计
    返回: all_responses列表，每个元素包含 max_step 和 step_counts
    """
    all_responses = []
    
    for sample in samples:
        step_maps = sample.get("step_map", [])
        for step_map in step_maps:
            if step_map and len(step_map) > 0:
                # 统计每个step出现的次数（即该step解码的token数）
                counter = Counter(step_map)
                max_step = max(step_map)
                all_responses.append({
                    "max_step": max_step,
                    "step_counts": counter
                })
    
    return all_responses


def compute_avg_token_counts(all_responses):
    """
    计算每个step的平均token count
    注意：只统计 max_step >= step 的response
    
    返回: steps列表, avg_counts列表, sample_counts列表
    """
    if not all_responses:
        return [], [], []
    
    global_max_step = max(r["max_step"] for r in all_responses)
    
    steps = []
    avg_counts = []
    sample_counts = []  # 每个step有多少个有效样本
    
    for step in range(1, global_max_step + 1):
        # 只统计 max_step >= step 的response
        valid_counts = [
            r["step_counts"].get(step, 0)
            for r in all_responses
            if r["max_step"] >= step
        ]
        
        if valid_counts:
            steps.append(step)
            avg_counts.append(sum(valid_counts) / len(valid_counts))
            sample_counts.append(len(valid_counts))
    
    return steps, avg_counts, sample_counts


def extract_title_from_filename(filename):
    """从文件名提取标题信息"""
    basename = os.path.basename(filename)
    # 移除前缀和后缀
    name = basename.replace("outputs-", "").replace(".json", "")
    return name


def plot_step_token_stats(steps, avg_counts, sample_counts, title, output_path):
    """绘制折线图"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 主Y轴：Avg. Parallel Token Count
    color1 = 'tab:blue'
    ax1.set_xlabel('Decoding Step', fontsize=12)
    ax1.set_ylabel('Avg. Parallel Token Count', color=color1, fontsize=12)
    line1 = ax1.plot(steps, avg_counts, color=color1, linewidth=1.5, label='Avg. Parallel Token Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # 次Y轴：有效样本数（可选，帮助理解数据）
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Number of Valid Samples', color=color2, fontsize=12)
    line2 = ax2.plot(steps, sample_counts, color=color2, linewidth=1, linestyle='--', alpha=0.7, label='Valid Samples')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标题
    plt.title(f'Step Token Statistics\n{title}', fontsize=14)
    
    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {output_path}")


def main():
    # 构建完整路径
    input_path = os.path.join(PROJECT_ROOT, INPUT_JSON)
    
    if OUTPUT_PNG is None:
        # 自动生成输出路径（与输入同目录）
        input_dir = os.path.dirname(input_path)
        input_basename = os.path.basename(input_path)
        output_basename = input_basename.replace(".json", "_step_stats.png")
        output_path = os.path.join(input_dir, output_basename)
    else:
        output_path = os.path.join(PROJECT_ROOT, OUTPUT_PNG)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"读取数据: {input_path}")
    
    # 1. 读取数据
    samples = load_data(input_path)
    print(f"共 {len(samples)} 个样本")
    
    # 2. 计算step统计
    all_responses = compute_step_stats(samples)
    print(f"共 {len(all_responses)} 个有效response")
    
    if not all_responses:
        print("错误: 没有找到有效的step_map数据")
        return
    
    # 3. 计算平均token count
    steps, avg_counts, sample_counts = compute_avg_token_counts(all_responses)
    
    global_max_step = max(r["max_step"] for r in all_responses)
    print(f"最大step: {global_max_step}")
    print(f"平均每步token数范围: {min(avg_counts):.2f} ~ {max(avg_counts):.2f}")
    
    # 4. 绘图
    title = extract_title_from_filename(INPUT_JSON)
    plot_step_token_stats(steps, avg_counts, sample_counts, title, output_path)


if __name__ == "__main__":
    main()
