from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum_tables_nskip200")
# @hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum")
def my_app(cfg: DictConfig) -> None:
    # print(cfg.run.seed)
    # return
    main(cfg)
    # print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()

# First activate the right virtual environment: source /home/tue/PycharmProjects/VirtualEnvironments/ConstrainedMD/bin/activate
# TO run this from command line use: PYTHONPATH=../ python train_pendulum.py from inside the example folder
