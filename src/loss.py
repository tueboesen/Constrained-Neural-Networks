import torch
import torch.nn.functional as F

from src.utils import LJ_potential
import torch
import torch.nn.functional as F

from src.utils import LJ_potential


def find_relevant_loss(loss_t,lossD_t,loss_v,lossD_v,use_validation,loss_fnc):
    """
    This function determines which loss function is the relevant to use.
    """
    if loss_fnc.lower() == 'mim':
        if use_validation:
            loss = lossD_v
        else:
            loss = lossD_t
    elif loss_fnc.lower() == 'eq':
        if use_validation:
            loss = loss_v
        else:
            loss = loss_t
    else:
        raise NotImplementedError("Loss function not implemented.")
    return loss

def loss_eq(x_pred, x_out, x_in):
    """
    Computes the relative MSE coordinate loss, which can be used by equivariant networks
    Note that loss and loss_ref should be divided by the batch size if you intend to use either of those numbers.
    assumes that x has the shape [particles,spatial dims]
    """
    # loss = torch.sum(torch.norm(x_pred - x_out, p=2, dim=1))
    # aa=torch.norm(x_pred - x_out, p=2, dim=1)
    loss = torch.mean(torch.sum((x_pred - x_out)**2,dim=1))
    loss_ref = torch.mean(torch.sum((x_in - x_out)**2,dim=1))
    # loss = F.mse_loss(x_pred, x_out)
    # loss_ref =
    # loss_ref = F.mse_loss(x_in, x_out)
    # loss_ref = torch.sum(torch.norm(x_in - x_out, p=2, dim=1))
    loss_rel = loss / loss_ref
    return loss, loss_ref, loss_rel

# def loss_eq(x_pred, x_out, x_in):
#     """
#     Computes the relative MSE coordinate loss, which can be used by equivariant networks
#     Note that loss and loss_ref should be divided by the batch size if you intend to use either of those numbers.
#     assumes that x has the shape [particles,spatial dims]
#     """
#     loss = torch.sum(torch.norm(x_pred - x_out, p=2, dim=1))
#     loss_ref = torch.sum(torch.norm(x_in - x_out, p=2, dim=1))
#     loss_rel = loss / loss_ref
#     return loss, loss_ref, loss_rel


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

def energy_momentum(r,v,m):
    """
    Computes the energy and momentum of the particles.
    This assumes that the velocity and position are given in sensible units, so remember to rescale them if they have been scaled!

    The Lennard Jones potential assumes that we are modelling water molecules, and leads to the potential energy. Each particle is assumed to be a single water molecule (9 coordinates
    """
    ECONV = 3.8087988458171926  # Converts from au*Angstrom^2/fs^2 to Hatree energy
    P = torch.sum((v.transpose(1, 2) @ m).norm(dim=1),dim=1)
    Ekin = torch.sum(0.5*m[...,0]*v.norm(dim=-1)**2, dim=1) * ECONV
    V_lj = LJ_potential(r)
    Epot = torch.sum(V_lj,dim=(1,2)) * ECONV
    E = Ekin + Epot
    return E, Ekin, Epot, P