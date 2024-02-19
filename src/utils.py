import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# from torch_cluster import radius_graph


def configuration_processor(c):
    """
    Processes the configuration file, adding dynamic variables, and sanity checking.
    The metadata file should be given a name that is canonical for that particular configuration
    """
    os.environ['MLFLOW_EXPERIMENT_NAME'] = c.run.experiment_name

    # initialize(config_path="conf", job_name="test_app")
    # cfg = compose(config_name="models")
    # c.model = getattr(cfg,f"model_{c.run.model_type}")
    if 'metafile' not in c.data:
        path = os.path.dirname(c.data.file)
        metapath = f"{path}/metadata/"
        c.data.metafile = os.path.join(metapath, f"split_{c.run.seed}_{c.data.n_train}_{c.data.n_val}_{c.data.n_test}_{c.data.nskip}.npz")
    if 'device' not in c.run:
        c.run.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # c.run.device = 'cpu'
        c.data.device = c.run.device

    var_lens = [len(val) for val in c.data.data_id.values()]
    if 'dim_in' in c.model.keys() and 'dim_in' not in c.model:
        c.model.dim_in = sum(var_lens)
    if 'name' not in c.run:
        c.run.name = f"{c.constraint.name}_{c.data.nskip}_{c.model.con_type}_{c.model.penalty_strength}_{c.model.regularization_strength}"
    # if 'irreps_inout' in c.model:
    #     c.model.irreps_inout = o3.Irreps(c.model.irreps_inout)
    # if 'irreps_hidden' in c.model:
    #     c.model.irreps_hidden = o3.Irreps(c.model.irreps_hidden)

    # c.run.loss_indices = c.data.data_id

    return c


def smooth_cutoff(x):
    """
    A smooth cutoff operation used to continuously bring a function from it variable down to 0 at a certain cutoff.
    """
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def convert_snapshots_to_future_state_dataset(n_skips, x):
    """
    converts snapshots of molecular dynamics data pairs of datapoints with n_skip+1 steps in between.
    """
    xout = x[1 + n_skips:]
    xin = x[:xout.shape[0]]
    return xin, xout


def fix_seed(seed, include_cuda=True):
    """
    Fixes the seed and enables a deterministic run, for replicable results.
    Note that the operating speed will be reduced.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def split_data(data, train_idx, val_idx, test_idx):
    """
    Splits data into training validation and testing
    """
    data_train = data[train_idx]
    data_val = data[val_idx]
    data_test = data[test_idx]
    return data_train, data_val, data_test


def atomic_masses(z):
    """
    A lookup table for atomic masses for the lightest elements
    z is the number of protons in the element.
    """
    atomic_masses = torch.tensor([0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948]).to(device=z.device)
    masses = atomic_masses[z.to(dtype=torch.int64)]
    return masses


def Distogram(r):
    """
    r should be of shape (n,3) or shape (nb,n,3)
    Note that it computes the distance squared, this is due to stability reasons in connection with autograd, if you want the actual distance, take the square-root
    """
    if r.ndim == 2:
        D = torch.relu(torch.sum(r.t() ** 2, dim=0, keepdim=True) + torch.sum(r.t() ** 2, dim=0, keepdim=True).t() - 2 * r @ r.t())
    elif r.ndim == 3:
        D = torch.relu(torch.sum(r.transpose(1, 2) ** 2, dim=1, keepdim=True) + torch.sum(r.transpose(1, 2) ** 2, dim=1, keepdim=True).transpose(1, 2) - 2 * r @ r.transpose(1, 2))
    else:
        raise Exception("shape not supported")

    return D


def LJ_potential(r, sigma=3.405, eps=119.8, rcut=8.4, Energy_conversion=1.5640976472642336e-06):
    """
    Lennard Jones potential
    """
    # eps_conv=31.453485691837958
    # V(r) = 4.0 * EPSILON * [(SIGMA / r) ^ 12 - (SIGMA / r) ^ 6]
    D = torch.sqrt(Distogram(r))
    M = D >= rcut
    Delta = sigma / D
    # Delta_au = Delta*5.291772E-1
    tmp = (Delta) ** 6
    V = 4.0 * eps * (tmp ** 2 - tmp)
    V[:, torch.arange(V.shape[-1]), torch.arange(V.shape[-1])] = 0
    V[M] = 0
    V *= Energy_conversion
    return V


def update_results_and_save_to_csv(results, epoch, loss_r_t, loss_v_t, cv_max_t, loss_r_v, loss_v_v, cv_max_v, MAEr_t, MAEr_v, MAEv_t, MAEv_v, csv_file, cv_t, cv_v, cv_energy_t, cv_energy_max_t,
                                   cv_energy_v, cv_energy_max_v):
    """
    Updates the results and saves it to a csv file.
    """
    result = pd.DataFrame({
        'epoch': [epoch],
        'loss_r_t': [loss_r_t],
        'loss_v_t': [loss_v_t],
        'loss_r_v': [loss_r_v],
        'loss_v_v': [loss_v_v],
        'cv_t': [cv_t],
        'cv_v': [cv_v],
        'cv_max_t': [cv_max_t],
        'cv_max_v': [cv_max_v],
        'cv_energy_t': [cv_energy_t],
        'cv_energy_v': [cv_energy_v],
        'cv_energy_max_t': [cv_energy_max_t],
        'cv_energy_max_v': [cv_energy_max_v],
        'MAE_r_t': [MAEr_t],
        'MAE_r_v': [MAEr_v],
        'MAE_v_t': [MAEv_t],
        'MAE_v_v': [MAEv_v],
    }, dtype=np.float32)
    result = result.astype({'epoch': np.int64})
    if epoch == 0:
        results = result
        results.iloc[-1:].to_csv(csv_file, header=True, sep='\t')
    else:
        results = pd.concat([results, pd.DataFrame.from_records(result)], ignore_index=True)
        results.iloc[-1:].to_csv(csv_file, mode='a', header=False, sep='\t')
    return results


def save_test_results_to_csv(loss_r, loss_v, cv_max, MAEr, MAEv, cv, cv_energy, cv_energy_max, csv_file):
    """
    Updates the results and saves it to a csv file.
    """
    result = pd.DataFrame({
        'loss_r': [loss_r],
        'loss_v': [loss_v],
        'cv': [cv],
        'cv_max': [cv_max],
        'cv_energy': [cv_energy],
        'cv_energy_max': [cv_energy_max],
        'MAE_r': [MAEr],
        'MAE_v': [MAEv],
    }, dtype=np.float32)
    results = result
    results.iloc[-1:].to_csv(csv_file, header=True, sep='\t')
    return results


def define_data_keys():
    base_keys = ['loss_r', 'loss_v', 'cv', 'cv_max', 'MAE_r', 'MAE_v']
    keys_r = [f"{key}_r" for key in base_keys]
    keys_v = [f"{key}_v" for key in base_keys]
    keys = keys_r + keys_v
    return keys
