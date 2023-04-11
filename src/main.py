import torch

from src.constraints import generate_constraints
from src.dataloader import load_data_wrapper
from src.loss import Loss
from src.networks import generate_neural_network
from src.optimization import optimize_model
from src.utils import fix_seed, configuration_processor

torch.set_printoptions(precision=10)

def main(c):
    """
    The main function which should be called with a fitting configuration.
    c is the input configuration, which dictates the run.
    """
    c = configuration_processor(c)
    torch.set_default_dtype(eval(c.run.precision))
    fix_seed(c.run.seed)  # Set a seed, so we make reproducible results.
    dataloaders = load_data_wrapper(c.data)
    con_fnc = generate_constraints(c.constraint)
    model = generate_neural_network(c.model,con_fnc=con_fnc)
    model.to(c.run.device)
    optimizer = torch.optim.Adam([{"params": model.params.base.parameters()},
                                  {"params": model.params.h.parameters()},
                                  {'params': model.params.close.parameters(), 'lr': c.optimizer.lr*0.1}], lr=c.optimizer.lr)
    loss_fnc = Loss(c.run.loss_indices,c.run.loss_type)
    loss = optimize_model(c,model,dataloaders,optimizer,loss_fnc)
    return loss
