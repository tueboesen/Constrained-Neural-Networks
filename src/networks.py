from src.network_mim import neural_network_mimetic


def generate_neural_network(c,model_type,con_fnc):
    if model_type == 'mim':
        model = neural_network_mimetic(con_fnc=con_fnc,**c)
    else:
        raise NotImplementedError(f"model type {model_type} not implemented yet.")
    return model