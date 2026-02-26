from huggingface_hub import snapshot_download

dataset_id = "dataset"
local_dir = "./PrimeIntellect"

snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",          
    local_dir=local_dir,
    local_dir_use_symlinks=False, # 下载实际文件
    resume_download=True          # 支持断点续传
)

print(f"数据集已下载至: {local_dir}")
