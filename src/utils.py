import torch
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_cluster import radius_graph
import math

def smooth_cutoff(x):
    """
    A smooth cutoff operation used to continuously bring a function from it variable down to 0 at a certain cutoff.
    """
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y

def convert_snapshots_to_future_state_dataset(n_skips,x):
    """
    converts snapshots of molecular dynamics data pairs of datapoints with n_skip+1 steps in between.
    """
    xout = x[1+n_skips:]
    xin = x[:xout.shape[0]]
    return xin,xout


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



def split_data(data,train_idx,val_idx,test_idx):
    """
    Splits data into training validation and testing
    """
    data_train = data[train_idx]
    data_val = data[val_idx]
    data_test = data[test_idx]
    return data_train,data_val,data_test


def atomic_masses(z):
    """
    A lookup table for atomic masses for the lightest elements
    z is the number of protons in the element.
    """
    atomic_masses = torch.tensor([0,1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085,30.974,32.06,35.45,39.948])
    masses = atomic_masses[z.to(dtype=torch.int64)]
    return masses.to(device=z.device)


def Distogram(r):
    """
    r should be of shape (n,3) or shape (nb,n,3)
    Note that it computes the distance squared, this is due to stability reasons in connection with autograd, if you want the actual distance, take the square-root
    """
    if r.ndim == 2:
        D = torch.relu(torch.sum(r.t() ** 2, dim=0, keepdim=True) + torch.sum(r.t() ** 2, dim=0, keepdim=True).t() - 2 * r @ r.t())
    elif r.ndim == 3:
        D = torch.relu(torch.sum(r.transpose(1,2) ** 2, dim=1, keepdim=True) + torch.sum(r.transpose(1,2) ** 2, dim=1, keepdim=True).transpose(1,2) - 2 * r @ r.transpose(1,2))
    else:
        raise Exception("shape not supported")

    return D




def LJ_potential(r, sigma=3.405,eps=119.8,rcut=8.4,Energy_conversion=1.5640976472642336e-06):
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

def update_results_and_save_to_csv(results,epoch,loss_r_t,loss_v_t,cv_max_t,loss_r_v,loss_v_v,cv_max_v,MAEr_t,MAEr_v,csv_file,cv_t,cv_v):
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
        'MAE_r_t': [MAEr_t],
        'MAE_r_v': [MAEr_v],
    }, dtype=np.float32)
    result = result.astype({'epoch': np.int64})
    if epoch == 0:
        results = result
        results.iloc[-1:].to_csv(csv_file, header=True, sep='\t')
    else:
        results = pd.concat([results, pd.DataFrame.from_records(result)], ignore_index=True)
        results.iloc[-1:].to_csv(csv_file, mode='a', header=False, sep='\t')
    return results


def run_model_MD_propagation_simulation(model, dataloader, max_radius=15,log=None,viz=None):
    """
    This function runs a Molecular dynamics propagation simulation using an already trained neural network.
    The dataloader should have been prepared with a dataset of similar future time stepping length as the model was originally trained on.
    This function propagates a particle system forward and continuously feeds it to itself, in order to propagate the MD simulation.
    """
    rscale = dataloader.dataset.rscale
    vscale = dataloader.dataset.vscale
    model.eval()
    nrep = len(dataloader.dataset)
    nsteps = dataloader.dataset.Rin.shape[1]

    lossD_his = torch.zeros(nrep,nsteps)
    loss_his = torch.zeros(nrep,nsteps)
    MAE_r_his = torch.zeros(nrep,nsteps)
    MAE_v_his = torch.zeros(nrep,nsteps)
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout, PEin, PEout, m) in enumerate(dataloader):
        nb, _,natoms, ndim = Rin.shape
        for j in range(nsteps):
            if j==0:
                Rin_vec = Rin[:,j].reshape(-1,Rin.shape[-1])
                Vin_vec = Vin[:,j].reshape(-1,Vin.shape[-1])
            else:
                Rin_vec = Rpred
                Vin_vec = Vpred

            Rout_vec = Rout[:, j].reshape(-1, Rout.shape[-1])
            Vout_vec = Vout[:, j].reshape(-1, Vout.shape[-1])
            x = torch.cat([Rin_vec, Vin_vec], dim=-1)
            z_vec = z.reshape(-1,z.shape[-1])
            batch = torch.arange(nb).repeat_interleave(natoms).to(device=Rin.device)
            edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=120)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            output = model(x, batch, z_vec, edge_src, edge_dst)
            Rpred = output[:, 0:ndim]
            Vpred = output[:, ndim:]

            loss_r, loss_ref_r, loss_rel_r = loss_eq(Rpred, Rout_vec, Rin_vec)
            loss_v, loss_ref_v, loss_rel_v = loss_eq(Vpred, Vout_vec, Vin_vec)
            loss_rel = (loss_rel_r + loss_rel_v) / 2

            lossD_r, lossD_ref_r, lossD_rel_r = loss_mim(Rpred, Rout_vec, Rin_vec, edge_src, edge_dst)
            lossD_v, lossD_ref_v, lossD_rel_v = loss_mim(Vpred, Vout_vec, Vin_vec, edge_src, edge_dst)
            lossD_rel = (lossD_rel_r + lossD_rel_v) / 2

            MAEr = torch.mean(torch.abs(Rpred - Rout_vec) * rscale).detach()
            MAEv = torch.mean(torch.abs(Vpred - Vout_vec) * vscale).detach()

            lossD_his[i,j] = lossD_rel.cpu().detach()
            loss_his[i,j] = loss_rel.cpu().detach()
            MAE_r_his[i,j] = MAEr.cpu().detach()
            MAE_v_his[i,j] = MAEv.cpu().detach()

    lossD_mean = lossD_his.mean(dim=0)
    loss_mean = loss_his.mean(dim=0)
    MAE_r_mean = MAE_r_his.mean(dim=0)
    MAE_v_mean = MAE_v_his.mean(dim=0)

    lossD_std = lossD_his.std(dim=0)
    loss_std = loss_his.std(dim=0)
    MAE_r_std = MAE_r_his.std(dim=0)
    MAE_v_std = MAE_v_his.std(dim=0)
    results_mean = [loss_mean,lossD_mean,MAE_r_mean,MAE_v_mean]
    results_std = [loss_std,lossD_std,MAE_r_std,MAE_v_std]
    outputfile = "{:}/endstep_results.torch".format(viz)
    data = {'loss':loss_his,'lossD':lossD_his,'MAE_r':MAE_r_his,'MAE_v':MAE_v_his}
    torch.save(data,outputfile)
    if viz is not None:
        nskip = dataloader.dataset.nskip
        x = torch.arange(nsteps)*(nskip+1)
        # fig, ax = plt.subplots()
        legends = ['loss','lossD','MAE_r','MAE_v']
        for ii,(legend,mean,std) in enumerate(zip(legends,results_mean,results_std)):
            fig, ax = plt.subplots(num=1, clear=True)
            pngfile = "{:}/endstep_{:}.png".format(viz,legend)
            ax.semilogy(x, mean, '-',label=legend)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.legend()
            plt.savefig(pngfile)
            plt.clf()

    return


def define_data_keys():
    base_keys = ['loss_r','loss_v','cv','cv_max','MAE_r','MAE_v']
    keys_r = [f"{key}_r" for key in base_keys]
    keys_v = [f"{key}_v" for key in base_keys]
    keys = keys_r + keys_v
    return keys

