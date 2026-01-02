import torch
import time
import multiprocessing
import sys

def occupy_gpu(gpu_id, target_mem_ratio=0.7, target_util_ratio=0.5):
    """
    子进程函数：控制单个GPU
    """
    try:
        # 在 spawn 模式下，子进程重新导入库，这里确保 CUDA 初始化
        device = torch.device(f"cuda:{gpu_id}")
        
        # --- 1. 显存占用 ---
        total_memory = torch.cuda.get_device_properties(device).total_memory
        occupy_memory = int(total_memory * target_mem_ratio)
        
        print(f"[GPU {gpu_id}] Total: {total_memory/1024**3:.2f}GB | Target: {occupy_memory/1024**3:.2f}GB")
        
        try:
            # 申请显存 (float32 = 4 bytes)
            num_elements = occupy_memory // 4
            memory_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        except RuntimeError as e:
            print(f"[GPU {gpu_id}] OOM Error: {e}")
            return

        # --- 2. 算力占用 (50%) ---
        size = 2000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        print(f"[GPU {gpu_id}] Running loop...")

        while True:
            start_time = time.time()
            
            # 计算工作 (Work)
            for _ in range(5):
                c = torch.matmul(a, b)
            torch.cuda.synchronize() # 等待计算完成
            
            end_time = time.time()
            work_time = end_time - start_time
            
            # 休眠 (Sleep)
            # work / (work + sleep) = 0.5  => sleep = work
            if target_util_ratio > 0 and target_util_ratio < 1:
                sleep_time = work_time * (1 - target_util_ratio) / target_util_ratio
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass # 静默退出
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")

def main():
    # 注意：获取 GPU 数量必须在 set_start_method 之后，
    # 或者像这里一样，先检查可用性，但不要在 spawn 前传递 CUDA 对象
    if not torch.cuda.is_available():
        print("No CUDA devices.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Mode: spawn")

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=occupy_gpu, args=(i, 0.7, 0.5))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    # --- 关键修复 ---
    # 必须在执行任何逻辑之前设置启动方式为 'spawn'
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # 防止多次设置报错
    
    main()