import torch

from src.constraints import generate_constraints
from src.dataloader import load_data_wrapper
from src.loss import Loss
from src.networks import generate_neural_network
from src.optimization import generate_constraints_minimizer
from src.training import optimize_model
from src.utils import fix_seed, configuration_processor

torch.set_printoptions(precision=10)


def main(c):
    """
    The main function which should be called with a fitting configuration.
    c is the input configuration, which dictates the run.
    """
    c = configuration_processor(c)
    torch.set_default_dtype(eval(c.run.precision))
    fix_seed(c.run.seed)
    dataloaders = load_data_wrapper(c.data)
    if 'scale' not in c.constraint:
        c.constraint.scale = dataloaders['train'].dataset.rscale
    con_fnc = generate_constraints(c.constraint)
    min_con_fnc = generate_constraints_minimizer(c.minimization,con_fnc)
    model = generate_neural_network(c.model, con_fnc=min_con_fnc)
    model.to(c.run.device)
    optimizer = torch.optim.Adam([{"params": model.params.base.parameters()},
                                  {"params": model.params.h.parameters()},
                                  {'params': model.params.close.parameters(), 'lr': c.optimizer.lr * 0.1}], lr=c.optimizer.lr)
    loss_fnc = Loss(c.run.loss_indices, c.run.loss_type)
    loss = optimize_model(c, model, dataloaders, optimizer, loss_fnc)
    return loss
