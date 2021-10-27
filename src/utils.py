import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
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



def split_data(data,train_idx,val_idx,test_idx):
    data_train = data[train_idx]
    data_val = data[val_idx]
    data_test = data[test_idx]
    return data_train,data_val,data_test


def atomic_masses(z):
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


def run_network_e3(model, dataloader, train, max_samples, optimizer, loss_fnc, batch_size=1, check_equivariance=False, max_radius=15, debug=False, log=None, viz=False):
    rscale = dataloader.dataset.rscale
    vscale = dataloader.dataset.vscale
    aloss = 0.0
    alossr = 0.0
    alossv = 0.0
    alossD = 0.0
    alossDr = 0.0
    alossDv = 0.0
    amomentum = 0.0
    aMAEr = 0.0
    aMAEv = 0.0
    P_hist = []
    E_hist = []
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout, PEin, PEout, m) in enumerate(dataloader):
        nb, natoms, ndim = Rin.shape
        ndims = Rin.shape[-1]
        optimizer.zero_grad()
        # Rin_vec = Rin.reshape(-1,Rin.shape[-1]*Rin.shape[-2])

        Rin_vec = Rin.reshape(-1,Rin.shape[-1])
        Rout_vec = Rout.reshape(-1,Rout.shape[-1])
        Vin_vec = Vin.reshape(-1,Vin.shape[-1])

        Vout_vec = Vout.reshape(-1,Vout.shape[-1])
        x = torch.cat([Rin_vec,Vin_vec],dim=-1)
        z_vec = z.reshape(-1,z.shape[-1])
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        edge_index = radius_graph(Rin_vec[:,:3], max_radius, batch, max_num_neighbors=120)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        output = model(x, batch, z_vec, edge_src, edge_dst)
        if output.isnan().any():
            output2 = model(x,batch, z_vec, edge_src, edge_dst)
        Rpred = output[:, 0:ndims]
        Vpred = output[:, ndims:]

        loss_r = torch.sum(torch.norm(Rpred.reshape(-1,3) - Rout_vec.reshape(-1,3), p=2, dim=1)) / nb
        loss_v = torch.sum(torch.norm(Vpred.reshape(-1,3) - Vout_vec.reshape(-1,3), p=2, dim=1)) / nb
        loss_abs = loss_r + loss_v
        loss_r_ref = torch.sum(torch.norm(Rin.reshape(-1,3) - Rout_vec.reshape(-1,3), p=2, dim=1)) / nb
        loss_v_ref = torch.sum(torch.norm(Vin.reshape(-1,3) - Vout_vec.reshape(-1,3), p=2, dim=1)) / nb
        loss_r_rel = loss_r / loss_r_ref
        loss_v_rel = loss_v / loss_v_ref
        loss_rel = (loss_r_rel + loss_v_rel)/2
        MAEr = torch.mean(torch.abs(Rpred - Rout_vec)*rscale).detach()
        MAEv = torch.mean(torch.abs(Vpred - Vout_vec)*vscale).detach()

        Vpred_real = Vpred.view(Vout.shape) * vscale
        Rpred_real = Rpred.view(Rout.shape) * rscale
        Vin_real = Vin * vscale
        Rin_real = Rin * rscale

        Econv = 3.8087988458171926 #Converts from au*Angstrom^2/fs^2 to Hatree energy

        # Ppred = torch.sum((Vpred_real.transpose(1, 2) @ m).norm(dim=1),dim=1) #Momentum is directional so do we take the correct magnitude here?
        # Ekin_pred = torch.sum(0.5*m[...,0]*Vpred_real.norm(dim=-1)**2, dim=1)
        # V_lj = LJ_potential(Rpred_real)
        # Epot_pred = torch.sum(V_lj,dim=(1,2))
        # Epred = (Ekin_pred + Epot_pred)*Econv
        # E_hist.append(Epred.detach())
        # P_hist.append(Ppred.detach())
        E_hist.append(torch.ones(5))
        P_hist.append(torch.ones(5))

        dRPred = torch.norm(Rpred[edge_src].reshape(-1,3) - Rpred[edge_dst].reshape(-1,3),p=2,dim=1)
        dRTrue = torch.norm(Rout_vec[edge_src].reshape(-1,3) - Rout_vec[edge_dst].reshape(-1,3),p=2,dim=1)
        dVPred = torch.norm(Vpred[edge_src].reshape(-1,3) - Vpred[edge_dst].reshape(-1,3),p=2,dim=1)
        dVTrue = torch.norm(Vout_vec[edge_src].reshape(-1,3) - Vout_vec[edge_dst].reshape(-1,3),p=2,dim=1)
        dRLast = torch.norm(Rin.reshape(Rout_vec.shape)[edge_src].reshape(-1,3) - Rin.reshape(Rout_vec.shape)[edge_dst].reshape(-1,3), p=2,dim=1)
        dVLast = torch.norm(Vin.reshape(Vout_vec.shape)[edge_src].reshape(-1,3) - Vin.reshape(Vout_vec.shape)[edge_dst].reshape(-1,3), p=2,dim=1)
        lossD_r_ref = F.mse_loss(dRLast,dRTrue)/nb
        lossD_v_ref = F.mse_loss(dVLast,dVTrue)/nb
        # lossD_rel = lossD_r_rel + lossD_v_rel
        lossD_r = F.mse_loss(dRPred,dRTrue)/nb
        lossD_v = F.mse_loss(dVPred,dVTrue)/nb
        lossD_r_rel = lossD_r / lossD_r_ref
        lossD_v_rel = lossD_v / lossD_v_ref
        # lossD_abs = lossD_r + lossD_v
        lossD_rel = (lossD_r_rel + lossD_v_rel)/2

        if check_equivariance:
            rot = o3.rand_matrix().to(device=x.device)
            Drot = model.irreps_in.D_from_matrix(rot)
            output_rot_after = output @ Drot
            output_rot = model(x @ Drot, batch, z_vec, edge_src, edge_dst)
            assert torch.allclose(output_rot,output_rot_after, rtol=1e-4, atol=1e-4)
            print("network is equivariant")
        if train:
            if loss_fnc == 'EQ':
                loss = loss_rel
                # loss = loss_v_rel
            else:
                loss = lossD_rel

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            if debug:
                weights = optimizer.param_groups[0]['params']
                weights_flat = [torch.flatten(weight) for weight in weights]
                weights_1d = torch.cat(weights_flat)
                assert not torch.isnan(weights_1d).any()
                assert not torch.isinf(weights_1d).any()
                print(f"{weights_1d.max()}, {weights_1d.min()}")

                grad_flat = [torch.flatten(weight.grad) for weight in weights]
                grad_1d = torch.cat(grad_flat)
                assert not torch.isnan(grad_1d).any()
                assert not torch.isinf(grad_1d).any()
                print(f"{grad_1d.max()}, {grad_1d.min()}")
            optimizer.step()

        if debug:
            log.debug(f"Loss={loss:2.2e}")
            Ekin_ref = torch.sum(0.5*m[...,0]*Vin_real.norm(dim=-1)**2, dim=1)
            V_lj = LJ_potential(Rin_real)
            Epot_ref = torch.sum(V_lj,dim=(1,2))
            Eref = Ekin_ref + Epot_ref
            Eref_conv = Eref * Econv
            torch.set_printoptions(precision=16)
            print(Eref_conv)



        aloss += loss_rel.detach()
        alossr += loss_r_rel.detach()
        alossv += loss_v_rel.detach()
        alossD += lossD_rel.detach()
        alossDr += lossD_r_rel.detach()
        alossDv += lossD_v_rel.detach()
        aMAEr += MAEr
        aMAEv += MAEv
        if (i + 1) * batch_size >= max_samples:
            break
    P_mean = torch.cat(P_hist).mean()
    E_std = torch.cat(E_hist).std()
    E_mean = torch.cat(E_hist).mean()
    E_rel_diff = (E_std / E_mean).abs()

    aloss /= (i + 1)
    alossr /= (i + 1)
    alossv /= (i + 1)
    alossD /= (i + 1)
    alossDr /= (i + 1)
    alossDv /= (i + 1)
    aMAEr /= (i + 1)
    aMAEv /= (i + 1)

    if viz:
        fig = plt.figure(num=1, figsize=[15, 10])
        plt.clf()
        axes = plt.axes(projection='3d')
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")

        rO_in = (Rin[0, :, 0:3] * rscale).cpu()
        rH1_in = (Rin[0, :, 3:6] * rscale).cpu()
        rH2_in = (Rin[0, :, 6:9] * rscale).cpu()

        rO_out = (Rout[0, :, 0:3] * rscale).cpu()
        rH1_out = (Rout[0, :, 3:6] * rscale).cpu()
        rH2_out = (Rout[0, :, 6:9] * rscale).cpu()

        axes.scatter(rO_out[:, 0], rO_out[:, 1], rO_out[:, 2], s=100, c='blue', depthshade=True)
        axes.scatter(rH1_out[:, 0], rH1_out[:, 1], rH1_out[:, 2], s=100, c='red', depthshade=True)
        axes.scatter(rH2_out[:, 0], rH2_out[:, 1], rH2_out[:, 2], s=100, c='red', depthshade=True)

        rO_pred = Rpred_real[0, :, 0:3].detach().cpu()
        rH1_pred = Rpred_real[0, :, 3:6].detach().cpu()
        rH2_pred = Rpred_real[0, :, 6:9].detach().cpu()

        axes.scatter(rO_pred[:, 0], rO_pred[:, 1], rO_pred[:, 2], s=100, c='lightblue', depthshade=True)
        axes.scatter(rH1_pred[:, 0], rH1_pred[:, 1], rH1_pred[:, 2], s=100, c='lightpink', depthshade=True)
        axes.scatter(rH2_pred[:, 0], rH2_pred[:, 1], rH2_pred[:, 2], s=100, c='lightpink', depthshade=True)

        axes.scatter(rO_in[:, 0], rO_in[:, 1], rO_in[:, 2], s=100, c='lightblue', depthshade=True)
        axes.scatter(rH1_in[:, 0], rH1_in[:, 1], rH1_in[:, 2], s=100, c='lightpink', depthshade=True)
        axes.scatter(rH2_in[:, 0], rH2_in[:, 1], rH2_in[:, 2], s=100, c='lightpink', depthshade=True)

        axes.quiver(rO_in[:, 0], rO_in[:, 1], rO_in[:, 2], rO_pred[:, 0] - rO_in[:, 0], rO_pred[:, 1] - rO_in[:, 1], rO_pred[:, 2] - rO_in[:, 2])
        axes.quiver(rO_pred[:, 0], rO_pred[:, 1], rO_pred[:, 2], rO_out[:, 0] - rO_pred[:, 0], rO_out[:, 1] - rO_pred[:, 1], rO_out[:, 2] - rO_pred[:, 2])

        filename = "{}fig.png".format(viz)
        fig.savefig(filename)

    return aloss, alossr, alossv, alossD, alossDr, alossDv, aMAEr, aMAEv, P_mean, E_rel_diff

def LJ_potential(r, sigma=3.405,eps=119.8,rcut=8.4,Energy_conversion=1.5640976472642336e-06):
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



def use_proteinmodel(model,dataloader,train,max_samples,optimizer, w=0.0, batch_size=1, reg=False):
    aloss = 0.0
    aloss_distogram_rel = 0.0
    aloss_distogram = 0.0
    aloss_coords = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (seq,batch, coords, M, edge_index,edge_index_all) in enumerate(dataloader):
        if torch.sum(M) < 5 or len(edge_index_all[0]) == 0:
            continue

        nb = len(torch.unique(batch))
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_src_all = edge_index_all[0]
        edge_dst_all = edge_index_all[1]

        optimizer.zero_grad()
        coords_init = 0.001*torch.ones_like(coords)
        coords_init[::2,::3] += 0.1
        coords_init = torch.cumsum(coords_init,dim=0)
        coords_pred = model(x=coords_init,batch=batch,node_attr=seq, edge_src=edge_src, edge_dst=edge_dst)
        dd = coords[1:, :3] - coords[:-1, :3]
        dn = torch.norm(dd,p=2,dim=1)
        dd2 = coords_init[1:,:3] - coords_init[:-1,:3]
        dn2 = torch.norm(dd2,p=2,dim=1)

        dRPred = torch.norm(coords_pred[edge_src_all] - coords_pred[edge_dst_all],p=2,dim=1)
        dRTrue = torch.norm(coords[edge_src_all] - coords[edge_dst_all],p=2,dim=1)
        lossD = F.mse_loss(dRPred,dRTrue)/nb
        lossD_rel = lossD / F.mse_loss(dRTrue * 0, dRTrue)

        t2 = time.time()
        # with profiler.record_function("backward"):
        if reg:
            con = model.constraints({'x': coords_pred, 'batch': batch,'z':seq})
            R = con['c']
            loss = lossD + R
        else:
            loss = lossD
        if train:
            loss.backward()
            optimizer.step()
        aloss += lossD.detach()
        aloss_distogram_rel += lossD_rel.detach()
        if (i + 1) * batch_size >= max_samples:
            break
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        if torch.isnan(aloss):
            print("problems")

    aloss /= (i + 1)
    aloss_distogram_rel /= (i + 1)

    return aloss, aloss_distogram_rel



def run_network_covid_e3(model, dataloader, train, max_samples, optimizer, batch_size=1, check_equivariance=False, max_radius=15, debug=True):
    aloss = 0.0
    alossr = 0.0
    alossv = 0.0
    alossD = 0.0
    alossDr = 0.0
    alossDv = 0.0
    aloss_ref = 0.0
    amomentum = 0.0
    aMAEr = 0.0
    aMAEv = 0.0
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

        edge_index = radius_graph(Rin_vec[:,0:3], max_radius, batch, max_num_neighbors=100)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        output = model(x, batch, z_vec, edge_src, edge_dst)
        Rpred = output[:, 0:9]
        Vpred = output[:, 9:]

        loss_r = torch.sum(torch.norm(Rpred - Rout_vec, p=2, dim=1)) / nb
        loss_v = torch.sum(torch.norm(Vpred - Vout_vec, p=2, dim=1)) / nb
        loss = loss_r + loss_v
        loss_r_last_step = torch.sum(torch.norm(Rin.reshape(Rout_vec.shape) - Rout_vec, p=2, dim=1)) / nb
        loss_v_last_step = torch.sum(torch.norm(Vin.reshape(Vout_vec.shape) - Vout_vec, p=2, dim=1)) / nb
        loss_last_step = loss_r_last_step + loss_v_last_step
        MAEr = torch.mean(torch.abs(Rpred - Rout_vec)).detach()
        MAEv = torch.mean(torch.abs(Vpred - Vout_vec)).detach()

        # dRPred = torch.norm(Rpred[edge_src] - Rpred[edge_dst],p=2,dim=1)
        # dRTrue = torch.norm(Rout_vec[edge_src] - Rout_vec[edge_dst],p=2,dim=1)
        # dVPred = torch.norm(Vpred[edge_src] - Vpred[edge_dst],p=2,dim=1)
        # dVTrue = torch.norm(Vout_vec[edge_src] - Vout_vec[edge_dst],p=2,dim=1)
        # lossD_r = F.mse_loss(dRPred,dRTrue)/nb
        # lossD_v = F.mse_loss(dVPred,dVTrue)/nb
        # lossD = lossD_r + lossD_v


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
        if debug:
            print("{:} Loss={:2.2e}".format(i,loss))

        aloss += loss.detach()
        alossr += loss_r.detach()
        alossv += loss_v.detach()
        # alossD += lossD.detach()
        # alossDr += lossD_r.detach()
        # alossDv += lossD_v.detach()
        # amomentum += Ppred.detach()
        aloss_ref += loss_last_step
        aMAEr += MAEr
        aMAEv += MAEv
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
    aMAEr /= (i + 1)
    aMAEv /= (i + 1)

    return aloss, alossr, alossv, alossD, alossDr, alossDv, aloss_ref, amomentum, aMAEr, aMAEv



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
