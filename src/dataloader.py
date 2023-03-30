import inspect
import math
import os
import time
from os.path import exists

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_cluster import radius_graph

from src.multibodypendulum.dataloader import load_multibodypendulum_data
from src.water.dataloader import load_water_data


def load_data_wrapper(c):
    """
    Wrapper function that handles the different supported data_types.
    """
    c_dict = OmegaConf.to_container(c)
    name = c_dict.pop("name", None)

    if name == 'multibodypendulum':
        dataloaders = load_multibodypendulum_data(**c_dict) # Standard method for loading data
    elif name == 'water':
        dataloaders = load_water_data(**c_dict)
    else:
        raise NotImplementedError(f"The data type {c.type} has not been implemented yet")

    # if 'rscale' not in c.data:
    #     c.data.rscale = rscale
    # if 'vscale' not in c.data:
    #     c.data.vscale = vscale
    return dataloaders
