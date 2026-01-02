#!/usr/bin/env python3
"""
AUP (Accuracy Under Parallelism) 计算脚本
在下方配置区域填入数据，然后运行脚本即可
"""
import math

# ============================================================
# 配置区域 - 在这里手动输入数据
# ============================================================

# 并行度列表 (TPF: tokens per forward)
RHO = [2.75,3.04,3.11,3.19]

# 对应的准确率列表 (百分比, 0-100)
Y = [91.56,90.57,90.32,89.89]

# 基准最大准确率 (通常是所有方法中的最高准确率)
Y_MAX = 66.9

# 可选参数
ALPHA = 3.0          # 惩罚因子 (默认 3.0)
Y_MIN_OFFSET = 5.0   # 最小准确率阈值偏移 (默认 5.0%)
IS_PRINT = True      # 是否打印详细计算过程

# ============================================================
# AUP 计算函数
# ============================================================

def weight_function(y: float, y_max: float, alpha: float = 3.0) -> float:
    """质量加权函数 W(y) = min(exp(-alpha * (1 - y/y_max)), 1)"""
    return min(math.exp(-alpha * (1 - y / y_max)), 1.0)

def get_aup(rho: list[float], y: list[float], y_max: float, alpha: float = 3.0, y_min_offset: float = 5.0, is_print: bool = False) -> float:
    """
    计算 Accuracy Under Parallelism (AUP) 指标
    
    参数:
        rho: 并行度列表 (TPF, tokens per forward)
        y: 准确率列表 [0, 100] (百分比)
        y_max: 所有方法中的最大准确率 (用于归一化)
        alpha: 准确率下降的惩罚因子 (默认: 3.0)
        y_min_offset: 最小准确率阈值偏移 (默认: 5.0, 即 5%)
    
    返回:
        AUP 分数 (标量值)
    """
    assert len(rho) == len(y), "rho and y must have the same length"
    assert len(rho) > 0, "rho and y must not be empty"
    assert all(r > 0 for r in rho), "all rho must be positive"
    
    # 检查 y 值是否在 [0, 100] 范围内
    if any(acc < 1.0 for acc in y):
        print("\033[91mWarning: Detected accuracy values < 1.0. Please check if accuracy should be in percentage (0-100) instead of (0-1).\033[0m")
    
    # 按 rho 排序
    sorted_pairs = sorted(zip(rho, y), key=lambda x: x[0])
    sorted_rho, sorted_y = zip(*sorted_pairs)
    sorted_rho, sorted_y = list(sorted_rho), list(sorted_y)
    
    # 按 y_min 阈值过滤 (y_1 - y_min_offset)
    y_1 = sorted_y[0]
    assert y_1 - sorted_y[-1] <= y_min_offset, f"Accuracy degradation is too large: minimum accuracy should be at least {y_min_offset:.2f} lower than the maximum accuracy. Max Acc: {y_1}, min Acc: {sorted_y[-1]}"
    y_min = y_1 - y_min_offset
    filtered_pairs = [(r, acc) for r, acc in zip(sorted_rho, sorted_y) if acc >= y_min]
    assert len(filtered_pairs) > 0, f"No valid pairs after filtering with y_min={y_min}"
    
    filtered_rho, filtered_y = zip(*filtered_pairs)
    filtered_rho, filtered_y = list(filtered_rho), list(filtered_y)
    
    # 计算 AUP: 第一项 + 梯形积分
    aup = filtered_rho[0] * filtered_y[0]
    formula_parts = [f"{filtered_rho[0]:.2f} * {filtered_y[0]:.2f}"]
    
    for i in range(1, len(filtered_rho)):
        y_i = filtered_y[i]
        y_prev = filtered_y[i-1]
        w_i = weight_function(y_i, y_max, alpha)
        w_prev = weight_function(y_prev, y_max, alpha)
        term = 0.5 * (filtered_rho[i] - filtered_rho[i-1]) * (y_i * w_i + y_prev * w_prev)
        aup += term
        formula_parts.append(f"0.5 * ({filtered_rho[i]:.2f}-{filtered_rho[i-1]:.2f}) * ({y_i:.2f}*{w_i:.4f} + {y_prev:.2f}*{w_prev:.4f})")

    if is_print:
        formula = "    AUP = " + " + ".join(formula_parts) + f" = {aup:.2f}"
        print(formula)
    
    return aup

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUP (Accuracy Under Parallelism) 计算")
    print("=" * 60)
    
    print("\n输入数据:")
    print(f"  并行度 ρ (rho): {RHO}")
    print(f"  准确率 y:       {Y}")
    print(f"  基准准确率 y_max: {Y_MAX}")
    print(f"  惩罚因子 α:     {ALPHA}")
    print(f"  阈值偏移:       {Y_MIN_OFFSET}%")
    
    print("\n" + "-" * 60)
    print("计算过程:")
    
    try:
        aup_score = get_aup(
            rho=RHO,
            y=Y,
            y_max=Y_MAX,
            alpha=ALPHA,
            y_min_offset=Y_MIN_OFFSET,
            is_print=IS_PRINT
        )
        
        print("-" * 60)
        print(f"\n结果: AUP = {aup_score:.4f}")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n错误: {e}")
        print("请检查输入数据是否符合要求")
