import torch
import torch.nn.functional as F

from src.utils import LJ_potential


def loss_wrapper(var_ids, loss_type):
    if loss_type == 'eq' or loss_type == 'mim':
        loss_fnc = Loss(var_ids, loss_type=loss_type)
    elif loss_type == 'mse':
        loss_fnc = F.mse_loss
    else:
        raise NotImplementedError(f"The loss {loss_type} has not been implemented yet.")
    return loss_fnc




class Loss:
    """
    A loss class that can handle both equivariant loss (where we do a coordinate loss using MSE), and mimetic loss (where we compute inter-particle distances).
    For both methods we compute the absolute loss, a reference loss, and the relative loss.
    The losses are computed for each group of indices given in var_ids and the average is taken.
    This enables us to compute the loss for both position and velocity separately.
    """

    def __init__(self, var_ids, loss_type):
        self.loss_type = loss_type
        self.var_ids = var_ids
        if loss_type == 'eq':
            loss_fnc = loss_eq
        elif loss_type == 'mim':
            loss_fnc = loss_mim
        else:
            raise NotImplementedError(f"The loss {loss_type} has not been implemented yet.")
        self.loss_fnc = loss_fnc

    def __call__(self, x_pred, x_out, x_in, edge_src, edge_dst, reduce=True):
        loss = 0
        for (var_name, idx) in self.var_ids.items():
            var_pred = x_pred[..., idx]
            var_out = x_out[..., idx]
            var_in = x_in[..., idx]
            loss_abs, loss_ref, loss_rel = self.loss_fnc(var_pred, var_out, var_in, edge_src, edge_dst, reduce=reduce)
            loss = loss + loss_rel
        n = len(self.var_ids)
        return loss / n




def loss_eq(x_pred, x_out, x_in, edge_src=None, edge_dst=None, reduce=True):
    """
    Computes the relative MSE coordinate loss, which can be used by equivariant networks
    Note that loss and loss_ref should be divided by the batch size if you intend to use either of those numbers.
    Assumes that x has the shape [particles,spatial dims]
    """
    if reduce:
        loss = torch.mean(torch.sum((x_pred - x_out) ** 2, dim=-1))
        loss_ref = torch.mean(torch.sum((x_in - x_out) ** 2, dim=-1))
    else:
        loss = torch.sum((x_pred - x_out) ** 2, dim=-1)
        loss_ref = torch.sum((x_in - x_out) ** 2, dim=-1)
    loss_rel = loss / loss_ref
    return loss, loss_ref, loss_rel


def loss_mim(x_pred, x_out, x_in, edge_src, edge_dst):
    """
    Computes the inter-particle MSE loss, this loss is rotational invariant and hence can be used by non-equivariant networks.
    Note that loss and loss_ref should be divided by the batch size if you intend to use either of those numbers.
    assumes that x has the shape [particles,spatial dims]

    edge_src and edge_dst are the edge connections between the various particles.
    """
    dx_pred = torch.norm(x_pred[edge_src] - x_pred[edge_dst], p=2, dim=1)
    dx_out = torch.norm(x_out[edge_src] - x_out[edge_dst], p=2, dim=1)
    dx_in = torch.norm(x_in[edge_src] - x_in[edge_dst], p=2, dim=1)

    loss = F.mse_loss(dx_pred, dx_out)
    loss_ref = F.mse_loss(dx_in, dx_out)
    loss_rel = loss / loss_ref
    return loss, loss_ref, loss_rel


def energy_momentum(r, v, m):
    """
    Computes the energy and momentum of the particles.
    This assumes that the velocity and position are given in sensible units, so remember to rescale them if they have been scaled!

    The Lennard Jones potential assumes that we are modelling water molecules, and leads to the potential energy. Each particle is assumed to be a single water molecule (9 coordinates
    """
    ECONV = 3.8087988458171926  # Converts from au*Angstrom^2/fs^2 to Hatree energy
    P = torch.sum((v.transpose(1, 2) @ m).norm(dim=1), dim=1)
    Ekin = torch.sum(0.5 * m[..., 0] * v.norm(dim=-1) ** 2, dim=1) * ECONV
    V_lj = LJ_potential(r)
    Epot = torch.sum(V_lj, dim=(1, 2)) * ECONV
    E = Ekin + Epot
    return E, Ekin, Epot, P
