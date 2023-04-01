import inspect
import math
import time

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch_cluster import radius_graph

from src.utils import atomic_masses, convert_snapshots_to_future_state_dataset
from src.npendulum import NPendulum, get_coordinates_from_angle, animate_pendulum
from src.vizualization import plot_water_snapshot


def load_data(file,data_type,device,nskip,n_train,n_val,n_test,use_val,use_test,batch_size, shuffle=True, use_endstep=False,file_val=None,model_specific=None):
    """
    Wrapper function that handles the different supported data_types.
    """
    if data_type == 'water':
        dataloader_train, dataloader_val, dataloader_test, dataloader_endstep =load_MD_data(file, data_type, device, nskip, n_train, n_val,n_test, use_val, use_test, batch_size, shuffle=shuffle, use_endstep=use_endstep)
    elif data_type == 'protein':
        dataloader_train= load_protein_data(file, data_type, device, n_train, batch_size, shuffle=shuffle)
        if use_val:
            dataloader_val = load_protein_data(file_val, data_type, device, n_val, batch_size, shuffle=shuffle)
        else:
            dataloader_val = None
        dataloader_endstep = None
        dataloader_test = None
    elif data_type == 'n-pendulum':
        dataloader_train, dataloader_val, dataloader_test =load_npendulum_data(data_type, device, nskip, n_train, n_val, n_test, use_val, use_test, batch_size, shuffle=shuffle,n=model_specific['n'],dt=model_specific['dt'],M=model_specific['M'],L=model_specific['L'], use_angles=model_specific['angles'],n_extra=model_specific['extra_simulation_steps'])
        dataloader_endstep = None
    else:
        NotImplementedError("The data_type={:} has not been implemented in function {:}".format(data_type, inspect.currentframe().f_code.co_name))
    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep


def load_npendulum_data(data_type,device,nskip,n_train,n_val,n_test,use_val,use_test,batch_size,n,dt, M, L, shuffle=True,n_extra=1000,use_angles=False):
    """
    Creates a multibody pendulum dataset and loads it into standard pytorch dataloaders.
    """
    def select_indices(idx,v_tuple,device):
        v_tuple_sel = []
        for vector in v_tuple:
            vector_sel = vector[idx].to(device)
            v_tuple_sel.append(vector_sel)
        return v_tuple_sel

    # assert len(L) == n
    # assert len(M) == n
    #
    # L = torch.tensor(L)
    # M = torch.tensor(M)
    # assert (L == 1).all(), f"Current implementation only supports npendulums with Lengths=1, you selected {L}"
    # assert (M == 1).all(), f"Current implementation only supports npendulums with Mass=1, you selected {M}"
    #
    # theta0 = 0.5*math.pi*torch.ones(n)
    # dtheta0 = 0.0*torch.ones(n)
    # nsteps = n_train + use_val * n_val + use_test * n_test + nskip + n_extra
    #
    # Npend = NPendulum(n,dt)
    #
    # t0 = time.time()
    # times, thetas, dthetas = Npend.simulate(nsteps,theta0,dtheta0)
    # t1 = time.time()
    # print(f"simulated {nsteps} steps for a {n}-pendulum in {t1-t0:2.2f}s")
    file = './../data/multibodypendulum/multibodypendulum.npz'
    with np.load(file) as data:
        theta = data['theta']
        dtheta = data['dtheta']
    thetas = torch.from_numpy(theta)
    dthetas = torch.from_numpy(dtheta)
    # import multibodypendulum as mbp
    # x,y,vx,vy = mbp.MultiBodyPendulum.get_coordinates_from_angles(theta,dtheta)
    #
    # R = torch.cat((x.T[:,None,:,None],y.T[:,None,:,None]),dim=-1)
    # V = torch.cat((vx.T[:,None,:,None],vy.T[:,None,:,None]),dim=-1)
    # Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    # Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    # particle_type = torch.ones_like(Rin).repeat(1,1,1,2) # we repeat it twice for R and V, which constitute our vector at the end
    # particle_mass = torch.ones_like(Rin).repeat(1,1,1,2)
    # # particle_type = torch.ones((Rin.shape[0],Rin.shape[1],Rin.shape[2]))
    # # particle_mass = torch.ones((Rin.shape[0],Rin.shape[1],Rin.shape[2]))
    #
    # features = {}
    # features['Rin'] = Rin.to(device)
    # features['Rout'] = Rout.to(device)
    # features['Vin'] = Vin.to(device)
    # features['Vout'] = Vout.to(device)
    # features['particle_type'] = particle_type.to(device)
    # features['particle_mass'] = particle_mass.to(device)
    # return features



    Ra = (thetas.clone().T)[:,:,None]
    Va = (dthetas.clone().T)[:,:,None]

    x,y,vx,vy = get_coordinates_from_angle(thetas,dthetas)
    # animate_pendulum(x.numpy(), y.numpy(),vx.numpy(),vy.numpy())

    x = x.T
    y = y.T
    vx = vx.T
    vy = vy.T
    R = torch.cat((x[:,1:,None],y[:,1:,None]),dim=2)
    V = torch.cat((vx[:,1:,None],vy[:,1:,None]),dim=2)

    # Vm = V.sum(dim=1)
    # Rm = R.sum(dim=1)
    # g = 9.82
    # v2 = vx**2 + vy**2
    # K = 0.5*torch.sum(v2[:,1:],dim=1)
    # P = g * torch.sum(y[:,1:],dim=1)
    # E = K + P
    # E0 = E.mean()


    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    Rina, Routa = convert_snapshots_to_future_state_dataset(nskip, Ra)
    Vina, Vouta = convert_snapshots_to_future_state_dataset(nskip, Va)

    ndata = Rout.shape[0]
    print(f'Number of data: {ndata}')
    assert n_train+n_val <= ndata, f"The number of datasamples: {ndata} is less than the number of training samples: {n_train} + the number of validation samples: {n_val}"

    metafile = './../data/multibodypendulum/metadata/split_1234_100_100_100_200.npz'
    with np.load(metafile) as mf:
        # ndata = mf['ndata']
        # assert len(features[list(features)[0]]) == ndata, "The number of data points in the dataset does not match the number in the metadata."
        train_idx = mf['idx_train']
        val_idx = mf['idx_val']
        test_idx = mf['idx_test']

    # ndata_rand = 0 + np.arange(ndata)
    # if shuffle:
    #     np.random.shuffle(ndata_rand)
    # train_idx = ndata_rand[:n_train]
    # val_idx = ndata_rand[n_train:n_train + n_val]
    # test_idx = ndata_rand[n_train+n_val:n_train + n_val+n_test]

    z = torch.arange(1,n+1)
    z = z.to(device)
    M = torch.tensor(M)
    M = M.to(device)

    v_tuple = (Rin,Rout,Vin,Vout,Rina,Routa,Vina,Vouta)
    Rin_sel,Rout_sel,Vin_sel,Vout_sel,Rina_sel,Routa_sel,Vina_sel,Vouta_sel = select_indices(train_idx,v_tuple,device)

    dataset_train = DatasetFutureState(data_type, Rin_sel, Rout_sel, z, Vin_sel, Vout_sel, m=M, device=device, Rin2=Rina_sel, Rout2=Routa_sel, Vin2=Vina_sel, Vout2=Vouta_sel)
    if use_angles:
        dataset_train.useprimary = False
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    if use_val:
        Rin_sel, Rout_sel, Vin_sel, Vout_sel, Rina_sel, Routa_sel, Vina_sel, Vouta_sel = select_indices(val_idx, v_tuple, device)
        dataset_val = DatasetFutureState(data_type, Rin_sel, Rout_sel, z, Vin_sel, Vout_sel, m=M, device=device, Rin2=Rina_sel, Rout2=Routa_sel, Vin2=Vina_sel, Vout2=Vouta_sel)
        if use_angles:
            dataset_val.useprimary = False
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size * 100, shuffle=False, drop_last=False)
    else:
        dataloader_val = None
    if use_test:
        Rin_sel, Rout_sel, Vin_sel, Vout_sel, Rina_sel, Routa_sel, Vina_sel, Vouta_sel = select_indices(test_idx, v_tuple, device)
        dataset_test = DatasetFutureState(data_type, Rin_sel, Rout_sel, z, Vin_sel, Vout_sel, m=M, device=device, Rin2=Rina_sel, Rout2=Routa_sel, Vin2=Vina_sel, Vout2=Vouta_sel)
        if use_angles:
            dataset_test.useprimary = False
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size * 100, shuffle=False, drop_last=False)
    else:
        dataloader_test = None
    return dataloader_train, dataloader_val, dataloader_test


def load_MD_data(file,data_type,device,nskip,n_train,n_val,n_test,use_val,use_test,batch_size, shuffle=True, use_endstep=False,viz_paper=False):
    """
    A function for loading molecular dynamics data
    """
    data = np.load(file)
    if 'R' in data.files:
        R = torch.from_numpy(data['R']).to(device=device,dtype=torch.get_default_dtype())
    else:
        R1 = torch.from_numpy(data['R1']).to(device=device,dtype=torch.get_default_dtype())
        R2 = torch.from_numpy(data['R2']).to(device=device,dtype=torch.get_default_dtype())
        R3 = torch.from_numpy(data['R3']).to(device=device,dtype=torch.get_default_dtype())
        R = torch.cat([R1,R2,R3],dim=2)
    z = torch.from_numpy(data['z']).to(device=device)
    if 'V' in data.files:
        V = torch.from_numpy(data['V']).to(device=device,dtype=torch.get_default_dtype())
    elif 'V1' in data.files:
        V1 = torch.from_numpy(data['V1']).to(device=device,dtype=torch.get_default_dtype())
        V2 = torch.from_numpy(data['V2']).to(device=device,dtype=torch.get_default_dtype())
        V3 = torch.from_numpy(data['V3']).to(device=device,dtype=torch.get_default_dtype())
        V = torch.cat([V1,V2,V3],dim=2)
    else: #Alternatively we use the positions to generate velocities, but we need to remove the first datasample for this to work
        V = R[1:] - R[:-1]
        R = R[1:]
    if 'F' in data.files:
        F = torch.from_numpy(data['F']).to(device=device,dtype=torch.get_default_dtype())
    else:
        F = None
    if 'KE' in data.files:
        KE = torch.from_numpy(data['KE']).to(device=device,dtype=torch.get_default_dtype())
    else:
        KE = None
    if 'PE' in data.files:
        PE = torch.from_numpy(data['PE']).to(device=device,dtype=torch.get_default_dtype())
    else:
        PE = None
    masses = atomic_masses(z)

    z = z.view(-1,3)

    # if viz_paper:
    #     plot_water_snapshot(R[0], filename=f"water_start.png")
    #     plot_water_snapshot(R[-1], filename=f"water_end.png")
    #     k = 50
    #     for ii in [0,2134,10001,43534,56534]:
    #         plot_water_snapshot(R[ii],R[ii+k],filename=f"water_{ii}_{k}.png")

    #We rescale the data
    Rscale = torch.sqrt(R.pow(2).mean())
    Vscale = torch.sqrt(V.pow(2).mean())
    R /= Rscale
    V /= Vscale

    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    Fin, Fout = convert_snapshots_to_future_state_dataset(nskip, F)
    KEin, KEout = convert_snapshots_to_future_state_dataset(nskip, KE)
    PEin, PEout = convert_snapshots_to_future_state_dataset(nskip, PE)

    R = None #This is a remnant from when I was working with very large datasets that could just barely fit in memory
    V = None
    F = None

    ndata = Rout.shape[0]
    natoms = z.shape[0]
    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    ndata_rand = 0 + np.arange(ndata)
    if shuffle:
        np.random.shuffle(ndata_rand)
    train_idx = ndata_rand[:n_train]
    val_idx = ndata_rand[n_train:n_train + n_val]
    test_idx = ndata_rand[n_train + n_val:n_train + n_val + n_test]
    endstep_idx = np.arange(10)

    Rin_train = Rin[train_idx]
    Rout_train = Rout[train_idx]
    Vin_train = Vin[train_idx]
    Vout_train = Vout[train_idx]
    Fin_train = Fin[train_idx]
    Fout_train = Fout[train_idx]
    KEin_train = KEin[train_idx]
    KEout_train = KEout[train_idx]
    PEin_train = PEin[train_idx]
    PEout_train = PEout[train_idx]

    if use_val:
        Rin_val = Rin[val_idx]
        Rout_val = Rout[val_idx]
        Vin_val = Vin[val_idx]
        Vout_val = Vout[val_idx]
        Fin_val = Fin[val_idx]
        Fout_val = Fout[val_idx]
        KEin_val = KEin[val_idx]
        KEout_val = KEout[val_idx]
        PEin_val = PEin[val_idx]
        PEout_val = PEout[val_idx]

    if use_test:
        Rin_test = Rin[test_idx]
        Rout_test = Rout[test_idx]
        Vin_test = Vin[test_idx]
        Vout_test = Vout[test_idx]
        Fin_test = Fin[test_idx]
        Fout_test = Fout[test_idx]
        KEin_test = KEin[test_idx]
        KEout_test = KEout[test_idx]
        PEin_test = PEin[test_idx]
        PEout_test = PEout[test_idx]

    if use_endstep:
        nrepeats = 10
        nsteps = int(np.ceil((Rin.shape[0]-nrepeats) / (nskip+1)))
        Rin_endstep = torch.empty((nrepeats,nsteps,Rin.shape[-2],Rin.shape[-1]))
        Rout_endstep = torch.empty((nrepeats,nsteps,Rout.shape[-2],Rout.shape[-1]))
        Vin_endstep = torch.empty((nrepeats,nsteps,Rin.shape[-2],Rin.shape[-1]))
        Vout_endstep = torch.empty((nrepeats,nsteps,Rout.shape[-2],Rout.shape[-1]))
        for i in range(nrepeats):
            endstep_idx = torch.arange(start=i, end=Rin.shape[0]-nrepeats, step=nskip+1,dtype=torch.int64)
            Rin_endstep[i,:,:,:] = Rin[endstep_idx]
            Rout_endstep[i,:,:,:] = Rout[endstep_idx]
            Vin_endstep[i,:,:,:] = Vin[endstep_idx]
            Vout_endstep[i,:,:,:] = Vout[endstep_idx]

    Fin = None
    Fout = None
    KEin = None
    KEout = None
    PEin = None
    PEout = None

    dataset_train = DatasetFutureState(data_type,Rin_train, Rout_train, z, Vin_train, Vout_train, Fin_train, Fout_train, KEin_train, KEout_train, PEin_train, PEout_train, masses,device=device, rscale=Rscale, vscale=Vscale)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if use_val:
        dataset_val = DatasetFutureState(data_type,Rin_val, Rout_val, z, Vin_val, Vout_val, Fin_val, Fout_val, KEin_val, KEout_val, PEin_val, PEout_val, masses,device=device, rscale=Rscale, vscale=Vscale)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size*10, shuffle=False, drop_last=False)
    else:
        dataloader_val = None


    if use_test:
        dataset_test = DatasetFutureState(data_type,Rin_test, Rout_test, z, Vin_test, Vout_test, Fin_test, Fout_test, KEin_test, KEout_test, PEin_test, PEout_test, masses,device=device, rscale=Rscale, vscale=Vscale)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size*10, shuffle=False, drop_last=False)
    else:
        dataloader_test = None

    if use_endstep:
        dataset_endstep = DatasetFutureState(data_type,Rin_endstep, Rout_endstep, z, Vin_endstep, Vout_endstep, m=masses,device=device, rscale=Rscale, vscale=Vscale,nskip=nskip)
        dataloader_endstep = DataLoader(dataset_endstep, batch_size=1, shuffle=False, drop_last=False)
    else:
        dataloader_endstep = None

    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep

class DatasetFutureState(data.Dataset):
    """
    A dataset type for future state predictions.
    """
    def __init__(self, data_type, Rin, Rout, z, Vin, Vout, Fin=None, Fout=None, KEin=None, KEout=None, PEin=None, PEout=None, m=None, device='cpu', rscale=1, vscale=1, nskip=1, pos_only=False,Rin2=None,Rout2=None,Vin2=None,Vout2=None):
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
        self.device = device
        self.rscale = rscale
        self.vscale = vscale
        self.nskip = nskip
        self.particles_pr_node = Rin.shape[-1] // 3
        self.data_type = data_type
        self.pos_only = pos_only

        self.Rin2 = Rin2
        self.Rout2 = Rout2
        self.Vin2 = Vin2
        self.Vout2 = Vout2
        self.useprimary = True

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
        device = self.device
        z = self.z[:,None]
        if self.useprimary:
            Rin = self.Rin[index]
            Rout = self.Rout[index]
            Vin = self.Vin[index]
            Vout = self.Vout[index]
        else:
            Rin = self.Rin2[index]
            Rout = self.Rout2[index]
            Vin = self.Vin2[index]
            Vout = self.Vout2[index]

        if self.m is not None:
            m = self.m[:,None]
        else:
            m = 0
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