from huggingface_hub import snapshot_download

dataset_id = "dataset"
local_dir = "./PrimeIntellect"

snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Download actual files
    resume_download=True,  # Support resumable downloads
)

print(f"Dataset downloaded to: {local_dir}")
