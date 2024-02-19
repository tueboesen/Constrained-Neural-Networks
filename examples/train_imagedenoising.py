import hydra
from omegaconf import DictConfig

from src.main import main


@hydra.main(version_base=None, config_path="./../config", config_name="imagedenoising_penalty_search")
def my_app(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    my_app()
