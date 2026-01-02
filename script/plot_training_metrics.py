# ============== 配置区（手动修改此处） ==============
# 单文件模式：指定单个jsonl文件
INPUT_JSONL = "rl_sdar_with_value/results/training_metrics_epoch1-10.jsonl"

# 多文件合并模式：使用glob模式匹配多个文件（设置后会忽略 INPUT_JSONL）
# 例如: "rl_sdar_with_value/results/training_metrics_epoch*.jsonl"
INPUT_PATTERN = None

# 输出目录，None则自动输出到输入文件同目录
OUTPUT_DIR = None
# ==================================================

import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# 获取脚本所在目录的上级目录（Fast-RL根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_metrics_from_file(jsonl_path):
    """读取单个jsonl文件，提取 type=step 的数据"""
    steps_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "step":
                    steps_data.append(record["data"])
            except json.JSONDecodeError:
                continue
    return steps_data


def load_metrics(jsonl_paths):
    """读取多个jsonl文件，合并所有 type=step 的数据"""
    all_data = []
    for path in jsonl_paths:
        data = load_metrics_from_file(path)
        all_data.extend(data)
    
    # 按 current_epoch 和 global_step 排序
    all_data.sort(key=lambda x: (x.get("epoch", 0), x.get("global_step", 0)))
    
    return all_data


def extract_series(data, key):
    """从数据中提取指定key的序列"""
    values = []
    steps = []
    cumulative_step = 0
    
    for i, record in enumerate(data):
        if key in record:
            # 使用累积step作为x轴
            cumulative_step += 1
            steps.append(cumulative_step)
            values.append(record[key])
    
    return steps, values


def plot_dual_axis(steps, y1, y2, label1, label2, title, output_path):
    """绘制双Y轴图"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左Y轴
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel(label1, color=color1, fontsize=12)
    line1 = ax1.plot(steps, y1, color=color1, linewidth=1, alpha=0.8, label=label1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # 右Y轴
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel(label2, color=color2, fontsize=12)
    line2 = ax2.plot(steps, y2, color=color2, linewidth=1, alpha=0.8, label=label2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标题和图例
    plt.title(title, fontsize=14)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {output_path}")


def plot_single(steps, values, ylabel, title, output_path, color='tab:blue'):
    """绘制单指标折线图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(steps, values, color=color, linewidth=1, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {output_path}")


def main():
    # 确定输入文件列表
    if INPUT_PATTERN is not None:
        pattern_path = os.path.join(PROJECT_ROOT, INPUT_PATTERN)
        input_files = sorted(glob.glob(pattern_path))
        if not input_files:
            print(f"错误: 没有找到匹配 {INPUT_PATTERN} 的文件")
            return
        print(f"多文件模式: 找到 {len(input_files)} 个文件")
        output_prefix = "training_metrics_all"
        # 输出到第一个文件的目录
        output_dir = os.path.dirname(input_files[0]) if OUTPUT_DIR is None else os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    else:
        input_path = os.path.join(PROJECT_ROOT, INPUT_JSONL)
        if not os.path.exists(input_path):
            print(f"错误: 文件不存在 {input_path}")
            return
        input_files = [input_path]
        print(f"单文件模式: {input_path}")
        # 从文件名提取前缀
        basename = os.path.basename(input_path).replace(".jsonl", "")
        output_prefix = basename
        output_dir = os.path.dirname(input_path) if OUTPUT_DIR is None else os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("读取数据...")
    data = load_metrics(input_files)
    print(f"共 {len(data)} 个训练步骤记录")
    
    if not data:
        print("错误: 没有找到训练步骤数据")
        return
    
    # 提取各指标序列
    steps_reward, reward_mean = extract_series(data, "reward_mean")
    steps_reward_std, reward_std = extract_series(data, "reward_std")
    steps_policy, policy_loss = extract_series(data, "policy_loss")
    steps_kl, kl_loss = extract_series(data, "kl_loss")
    steps_ratio, ratio_mean = extract_series(data, "ratio_mean")
    steps_clip, clip_frac = extract_series(data, "clip_frac")
    
    # 图1: reward_std
    if reward_std:
        plot_single(
            steps_reward_std, reward_std,
            "Reward Std",
            "Reward Standard Deviation over Training",
            os.path.join(output_dir, f"{output_prefix}_reward_std.png"),
            color='tab:green'
        )
    
    # 图2: policy_loss
    if policy_loss:
        plot_single(
            steps_policy, policy_loss,
            "Policy Loss",
            "Policy Loss over Training",
            os.path.join(output_dir, f"{output_prefix}_policy_loss.png"),
            color='tab:red'
        )
    
    # 图3: kl_loss
    if kl_loss:
        plot_single(
            steps_kl, kl_loss,
            "KL Loss",
            "KL Loss over Training",
            os.path.join(output_dir, f"{output_prefix}_kl_loss.png"),
            color='tab:purple'
        )
    
    # 图4: ratio_mean
    if ratio_mean:
        plot_single(
            steps_ratio, ratio_mean,
            "Ratio Mean",
            "Importance Sampling Ratio Mean over Training",
            os.path.join(output_dir, f"{output_prefix}_ratio_mean.png"),
            color='tab:cyan'
        )
    
    # 图5: clip_frac
    if clip_frac:
        plot_single(
            steps_clip, clip_frac,
            "Clip Fraction",
            "Clip Fraction over Training",
            os.path.join(output_dir, f"{output_prefix}_clip_frac.png"),
            color='tab:brown'
        )
    
    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
