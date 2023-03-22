import os
import time
from datetime import datetime

import torch

from src import log
from src.constraints import generate_constraints
from src.dataloader import load_data_wrapper
from src.log import log_all_parameters, close_logger
from src.loss import find_relevant_loss, generate_loss_fnc
from src.network_e3 import neural_network_equivariant
# from src.network_eq_simple import neural_network_equivariant_simple
from src.network_mim import neural_network_mimetic
from src.networks import generate_neural_network
from src.optimization import run_model, optimize_model
from src.project_uplift import ProjectUpliftEQ
from src.utils import fix_seed, update_results_and_save_to_csv, run_model_MD_propagation_simulation, save_test_results_to_csv, configuration_processor
from src.vizualization import plot_training_and_validation
torch.set_printoptions(precision=10)

def main(c):
    """
    The main function which should be called with a fitting configuration dictionary.
    c is the input configuration dictionary, which dictates the run.
    """
    c = configuration_processor(c)
    torch.set_default_dtype(eval(c.run.precision))
    fix_seed(c.run.seed)  # Set a seed, so we make reproducible results.
    dataloaders = load_data_wrapper(c.data)
    con_fnc = generate_constraints(c.constraints,c.data.rscale,c.data.vscale)
    model = generate_neural_network(c.model,c.run.model_type,con_fnc=con_fnc)

    # Load previous model?

    model.to(c.run.device)
    optimizer = torch.optim.Adam([{"params": model.params.base.parameters()},
                                  {"params": model.params.h.parameters()},
                                  {'params': model.params.close.parameters(), 'lr': c.optimizer.lr*0.1}], lr=c.optimizer.lr)
    loss_fnc = generate_loss_fnc(c.run)
    optimize_model(c.run,model,dataloaders,optimizer,loss_fnc)
    return
