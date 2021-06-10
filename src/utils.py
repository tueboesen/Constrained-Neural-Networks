import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import random
import math

from e3nn import o3
from torch_cluster import radius_graph


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def smooth_cutoff2(x,cutoff_start=0.5,cutoff_end=1):
    '''
    x should be a vector of numbers that needs to smoothly be cutoff at 1
    :param x:
    :return:
    '''

    M1 = x < cutoff_end
    M2 = x > cutoff_start
    M_cutoff_region = M1 * M2
    M_out = x > 1
    s = torch.ones_like(x)
    s[M_out] = 0
    pi = math.pi
    s[M_cutoff_region] = 0.5 * torch.cos(pi * (x[M_cutoff_region]-cutoff_start) / (cutoff_end-cutoff_start)) + 0.5
    return s

def convert_snapshots_to_future_state_dataset(n_skips,x):
    """
    :param n_input_samples:
    :param n_skips:
    :param x:
    :return:
    """
    xout = x[1+n_skips:]
    xin = x[:xout.shape[0]]
    return xin,xout


def fix_seed(seed, include_cuda=True):
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


class DatasetFutureState(data.Dataset):
    def __init__(self, Rin, Rout, z, Vin, Vout, Fin=None, Fout=None, KEin=None, KEout=None, PEin=None, PEout=None, m=None):
        self.Rin = Rin
        self.Rout = Rout
        self.z = z
        self.Vin = Vin
        self.Vout = Vout
        self.Fin = Fin
        self.Fout = Fout
        self.KEin = KEin
        self.KEout = KEout
        self.PEin = PEin
        self.PEout = PEout
        self.m = m
        if Fin is None or Fout is None:
            self.useF = False
        else:
            self.useF = True
        if KEin is None or KEout is None:
            self.useKE = False
        else:
            self.useKE = True
        if PEin is None or PEout is None:
            self.usePE = False
        else:
            self.usePE = True
        return

    def __getitem__(self, index):
        Rin = self.Rin[index]
        Rout = self.Rout[index]
        z = self.z[:,None]
        Vin = self.Vin[index]
        Vout = self.Vout[index]
        m = self.m[:,None]
        if self.useF:
            Fin = self.Fin[index]
            Fout = self.Fout[index]
        else:
            Fin = 0
            Fout = 0
        if self.useKE:
            KEin = self.KEin[index]
            KEout = self.KEout[index]
        else:
            KEin = 0
            KEout = 0
        if self.usePE:
            PEin = self.PEin[index]
            PEout = self.PEout[index]
        else:
            PEin = 0
            PEout = 0
        return Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout,PEin, PEout,m

    def __len__(self):
        return len(self.Rin)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'

def atomic_masses(z):
    atomic_masses = torch.tensor([0,1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085])
    masses = atomic_masses[z.to(dtype=torch.int64)]
    return masses.to(device=z.device)


def Distogram(r):
    """
    r should be of shape (n,3)
    """
    D = torch.relu(torch.sum(r.t() ** 2, dim=0, keepdim=True) + torch.sum(r.t() ** 2, dim=0, keepdim=True).t() - 2 * r @ r.t())
    return D

def run_network_e3(model, dataloader, train, max_samples, optimizer, batch_size=1, check_equivariance=False, max_radius=15):
    aloss = 0.0
    alossr = 0.0
    alossv = 0.0
    alossD = 0.0
    alossDr = 0.0
    alossDv = 0.0
    aloss_ref = 0.0
    amomentum = 0.0
    MAE = 0.0
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout, PEin, PEout, m) in enumerate(dataloader):
        nb, natoms, ndim = Rin.shape
        optimizer.zero_grad()
        # Rin_vec = Rin.reshape(-1,Rin.shape[-1]*Rin.shape[-2])

        Rin_vec = Rin.reshape(-1,Rin.shape[-1])
        Rout_vec = Rout.reshape(-1,Rout.shape[-1])
        Vin_vec = Vin.reshape(-1,Vin.shape[-1])

        Vout_vec = Vout.reshape(-1,Vout.shape[-1])
        x = torch.cat([Rin_vec,Vin_vec],dim=-1)
        z_vec = z.reshape(-1,z.shape[-1])
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=100)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        output = model(x, batch, z_vec, edge_src, edge_dst)
        Rpred = output[:, 0:3]
        Vpred = output[:, 3:]

        loss_r = torch.sum(torch.norm(Rpred - Rout_vec, p=2, dim=1)) / nb
        loss_v = torch.sum(torch.norm(Vpred - Vout_vec, p=2, dim=1)) / nb
        loss = loss_r + loss_v
        loss_last_step = torch.sum(torch.norm(Rin.reshape(Rout_vec.shape) - Rout_vec, p=2, dim=1)) / nb
        MAEi = torch.mean(torch.abs(Rpred - Rout_vec)).detach()

        Ppred = torch.sum((Vpred.view(Vout.shape).transpose(1, 2) @ m).norm(dim=1)) / nb

        dRPred = torch.norm(Rpred[edge_src] - Rpred[edge_dst],p=2,dim=1)
        dRTrue = torch.norm(Rout_vec[edge_src] - Rout_vec[edge_dst],p=2,dim=1)
        dVPred = torch.norm(Vpred[edge_src] - Vpred[edge_dst],p=2,dim=1)
        dVTrue = torch.norm(Vout_vec[edge_src] - Vout_vec[edge_dst],p=2,dim=1)
        lossD_r = F.mse_loss(dRPred,dRTrue)/nb
        lossD_v = F.mse_loss(dVPred,dVTrue)/nb
        lossD = lossD_r + lossD_v


        if check_equivariance:
            rot = o3.rand_matrix().to(device=x.device)
            Drot = model.irreps_in.D_from_matrix(rot)
            output_rot_after = output @ Drot
            output_rot = model(x @ Drot, batch, z_vec, edge_src, edge_dst)
            assert torch.allclose(output_rot,output_rot_after, rtol=1e-4, atol=1e-4)
            print("network is equivariant")
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        alossr += loss_r.detach()
        alossv += loss_v.detach()
        alossD += lossD.detach()
        alossDr += lossD_r.detach()
        alossDv += lossD_v.detach()
        amomentum += Ppred.detach()
        aloss_ref += loss_last_step
        MAE += MAEi
        if (i + 1) * batch_size >= max_samples:
            break
    aloss /= (i + 1)
    alossr /= (i + 1)
    alossv /= (i + 1)
    alossD /= (i + 1)
    alossDr /= (i + 1)
    alossDv /= (i + 1)
    aloss_ref /= (i + 1)
    amomentum /= (i + 1)
    MAE /= (i + 1)

    return aloss, alossr, alossv, alossD, alossDr, alossDv, aloss_ref, amomentum, MAE


def run_network_eq(model,dataloader,train,max_samples,optimizer,batch_size=1,check_equivariance=False):
    aloss = 0.0
    aloss_ref = 0.0
    MAE = 0.0
    t_dataload = 0.0
    t_prepare = 0.0
    t_model = 0.0
    t_backprop = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, Ein, Eout) in enumerate(dataloader):
        nb, natoms, nhist, ndim = Rin.shape
        t0 = time.time()
        # Rin_vec = Rin.reshape(-1,Rin.shape[-1]*Rin.shape[-2])
        Rin_vec = Rin.reshape(nb*natoms,-1,3).transpose(1,2)
        Rout_vec = Rout.reshape(nb*natoms,3)
        z_vec = z.reshape(-1,z.shape[-1])
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        optimizer.zero_grad()
        t1 = time.time()

        output = model(Rin_vec,z_vec,batch)
        Rpred = output[:,:,-1]
        Vpred = output[:,:,0]
        t2 = time.time()

        loss = torch.sum(torch.norm(Rpred-Rout_vec,p=2,dim=1))/nb
        loss_last_step = torch.sum(torch.norm(Rin[:,:,-1,:].reshape(Rout_vec.shape) - Rout_vec, p=2,dim=1))/nb
        MAEi = torch.mean(torch.abs(Rpred - Rout_vec)).detach()

        if check_equivariance:
            rot = o3.rand_matrix(1)
            Rin_vec_rotated = (Rin_vec.transpose(1, 2) @ rot).transpose(1, 2)
            Rpred_rotated = model(Rin_vec_rotated,z_vec,batch)
            Rpred_rotated_after = (Rpred.transpose(1,2) @ rot).transpose(1,2)
            assert torch.allclose(Rpred_rotated, Rpred_rotated_after, rtol=1e-4, atol=1e-4)

        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_ref += loss_last_step
        MAE += MAEi
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    aloss_ref /= (i+1)
    MAE /= (i+1)
    t_dataload /= (i+1)
    t_prepare /= (i+1)
    t_model /= (i+1)
    t_backprop /= (i+1)

    return aloss, aloss_ref, MAE, t_dataload, t_prepare, t_model, t_backprop

def run_network(model,dataloader,train,max_samples,optimizer,batch_size=1,check_equivariance=False, max_radius=15):
    aloss = 0.0
    alossr = 0.0
    alossv = 0.0
    ap = 0.0
    ap_ref = 0.0
    aloss_ref = 0.0
    MAE = 0.0
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout, PEin, PEout, m) in enumerate(dataloader):
        nb, natoms, ndim = Rin.shape
        optimizer.zero_grad()

        # Rin_vec = Rin.reshape(-1,Rin.shape[-1]*Rin.shape[-2])
        Rin_vec = Rin.reshape(nb*natoms,-1)
        Vin_vec = Vin.reshape(nb*natoms,-1)
        Rout_vec = Rout.reshape(nb*natoms,-1)
        Vout_vec = Vout.reshape(nb*natoms,-1)
        z_vec = z.reshape(-1,z.shape[-1])
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)
        edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=100)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        optimizer.zero_grad()
        x = torch.cat([Rin_vec,Vin_vec],dim=-1)
        output = model(x, batch, z_vec,edge_src,edge_dst)
        Rpred = output[:,0:3]
        Vpred = output[:,3:]
        t2 = time.time()

        dRPred = torch.norm(Rpred[edge_src] - Rpred[edge_dst],p=2,dim=1)
        dRTrue = torch.norm(Rout_vec[edge_src] - Rout_vec[edge_dst],p=2,dim=1)
        RLast = Rin.view(Rout_vec.shape)
        dRLast = torch.norm(RLast[edge_src] - RLast[edge_dst],p=2,dim=1)
        dVPred = torch.norm(Vpred[edge_src] - Vpred[edge_dst],p=2,dim=1)
        dVTrue = torch.norm(Vout_vec[edge_src] - Vout_vec[edge_dst],p=2,dim=1)

        loss_last_step = F.mse_loss(dRLast, dRTrue)/nb
        loss_r = F.mse_loss(dRPred,dRTrue)/nb
        loss_v = F.mse_loss(dVPred,dVTrue)/nb
        loss = loss_r + loss_v
        loss = loss / loss_last_step
        MAEi = torch.mean(torch.abs(Rpred - Rout_vec)).detach()

        Ppred = torch.sum((Vpred.view(Vout.shape).transpose(1,2) @ m).norm(dim=1)) / nb
        Ptrue = torch.sum((Vout.transpose(1,2) @ m).norm(dim=1)) / nb


        if check_equivariance:
            rot = o3.rand_matrix(1)
            Rin_vec_rotated = (Rin_vec.transpose(1, 2) @ rot).transpose(1, 2)
            Rpred_rotated = model(Rin_vec_rotated,z_vec,batch)
            Rpred_rotated_after = (Rpred.transpose(1,2) @ rot).transpose(1,2)
            assert torch.allclose(Rpred_rotated, Rpred_rotated_after, rtol=1e-4, atol=1e-4)

        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        alossr += loss_r.detach()
        alossv += loss_v.detach()
        ap += Ppred.detach()
        ap_ref += Ptrue.detach()
        aloss_ref += loss_last_step
        MAE += MAEi

        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    alossr /= (i+1)
    alossv /= (i+1)
    aloss_ref /= (i+1)
    ap /= (i+1)
    ap_ref /= (i+1)
    MAE /= (i+1)

    return aloss, alossr, alossv, aloss_ref, ap, ap_ref, MAE

#
# def compute_inverse_square_distogram(r):
#     D2 = torch.relu(torch.sum(r ** 2, dim=1, keepdim=True) + \
#                    torch.sum(r ** 2, dim=1, keepdim=True).transpose(1,2) - \
#                    2 * r.transpose(1,2) @ r)
#     iD2 = 1 / D2
#     tmp = iD2.diagonal(0,dim1=1,dim2=2)
#     tmp[:] = 0
#     return D2, iD2
#
#
# def compute_graph(r,nn=15):
#     D2, iD2 = compute_inverse_square_distogram(r)
#
#     _, J = torch.topk(iD2, k=nn-1, dim=-1)
#     I = (torch.ger(torch.arange(nn), torch.ones(nn-1, dtype=torch.long))[None,:,:]).repeat(nb,1,1).to(device=z.device)
#     I = I.view(nb,-1)
#     J = J.view(nb,-1)
