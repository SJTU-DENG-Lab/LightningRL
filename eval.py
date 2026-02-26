import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()

    # Set GPU devices from config
    if hasattr(config.experiment, 'gpu_ids') and config.experiment.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.experiment.gpu_ids)
        print(f"Setting CUDA_VISIBLE_DEVICES={config.experiment.gpu_ids}")

    project_name = config.experiment.project
    eval_type = config.dataset.data_type

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base):
        cprint(f"This is sampling.", color = "green")
        subprocess.run(
            f'python sample.py '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='sample',
            check=True,
        )
    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        subprocess.run(
            f'python reward.py '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute():
        cprint(f"This is the execution.", color = "green")
        subprocess.run(
            f'python execute.py '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    
    sample(config.model_base)
    if eval_type == "code":
        execute()
    
    reward()




