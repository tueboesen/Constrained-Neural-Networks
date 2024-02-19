from omegaconf import OmegaConf

from src.imagedenoising.dataloader import load_imagedenoising_data
from src.multibodypendulum.dataloader import load_multibodypendulum_data
from src.water.dataloader import load_water_data


def load_data_wrapper(c):
    """
    Wrapper function that handles the different supported data_types.
    """
    c_dict = OmegaConf.to_container(c)
    name = c_dict.pop("name", None)

    if name == 'multibodypendulum':
        dataloaders = load_multibodypendulum_data(**c_dict)
    elif name == 'water':
        dataloaders = load_water_data(**c_dict)
    elif name == 'imagedenoising':
        dataloaders = load_imagedenoising_data(**c_dict)
    else:
        raise NotImplementedError(f"The data type {c.type} has not been implemented yet")
    return dataloaders
