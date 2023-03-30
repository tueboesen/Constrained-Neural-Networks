from omegaconf import OmegaConf

from src.network_e3 import neural_network_equivariant
from src.network_mim import neural_network_mimetic


def generate_neural_network(c,con_fnc):
    c_dict = OmegaConf.to_container(c)
    name = c_dict.pop("name", None)

    if name == 'mimetic':
        model = neural_network_mimetic(con_fnc=con_fnc,**c_dict)
    elif name == 'equivariant':
        model = neural_network_equivariant(con_fnc=con_fnc,**c_dict)
    else:
        raise NotImplementedError(f"model name {name} not implemented yet.")

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters {:}'.format(total_params))

    return model