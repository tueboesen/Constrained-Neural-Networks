import os
from hydra import compose, initialize
from src.main import main


def test_all():
    """
    A quick test that tests all yaml files located in the base config folder and checks whether they can run and returns a sensible loss.
    """
    path = './../config'
    for filename in os.listdir(path):
        if filename.endswith(".yaml"):
            with initialize(version_base=None, config_path="./../config"):
                cfg = compose(config_name=filename)
                cfg.run.epochs = 1
                loss = main(cfg)
                assert loss > 0
