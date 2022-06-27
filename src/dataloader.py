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


def load_data(file,data_type,device,nskip,n_train,n_val,use_val,use_test,batch_size, shuffle=True, use_endstep=False,file_val=None,model_specific=None):
    """
    Wrapper function that handles the different supported data_types.
    """
    if data_type == 'water':
        dataloader_train, dataloader_val, dataloader_test, dataloader_endstep =load_MD_data(file, data_type, device, nskip, n_train, n_val, use_val, use_test, batch_size, shuffle=shuffle, use_endstep=use_endstep)
    elif data_type == 'protein':
        dataloader_train= load_protein_data(file, data_type, device, n_train, batch_size, shuffle=shuffle)
        if use_val:
            dataloader_val = load_protein_data(file_val, data_type, device, n_val, batch_size, shuffle=shuffle)
        else:
            dataloader_val = None
        dataloader_endstep = None
        dataloader_test = None
    elif data_type == 'double-pendulum':
        dataloader_train, dataloader_val, =load_pendulum_data(file, data_type, device, nskip, n_train, n_val, use_val, use_test, batch_size, shuffle=shuffle)
        dataloader_test, dataloader_endstep = None, None
    elif data_type == 'n-pendulum':
        dataloader_train, dataloader_val, =load_npendulum_data(data_type, device, nskip, n_train, n_val, use_val, batch_size, shuffle=shuffle,n=model_specific['n'],dt=model_specific['dt'],M=model_specific['M'],L=model_specific['L'], use_angles=model_specific['angles'])
        dataloader_test, dataloader_endstep = None, None
    else:
        NotImplementedError("The data_type={:} has not been implemented in function {:}".format(data_type, inspect.currentframe().f_code.co_name))
    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep




def load_npendulum_data(data_type,device,nskip,n_train,n_val,use_val,batch_size,n,dt, M, L, shuffle=True,n_extra=1000,use_angles=False):
    assert len(L) == n
    assert len(M) == n

    L = torch.tensor(L)
    M = torch.tensor(M)
    assert (L == 1).all(), f"Current implementation only supports npendulums with Lengths=1, you selected {L}"
    assert (M == 1).all(), f"Current implementation only supports npendulums with Mass=1, you selected {M}"

    theta0 = 0.5*math.pi*torch.ones(n)
    dtheta0 = 0.0*torch.ones(n)
    nsteps = n_train + use_val *n_val+nskip + n_extra

    Npend = NPendulum(n,dt)

    t0 = time.time()
    times, thetas, dthetas = Npend.simulate(nsteps,theta0,dtheta0)
    t1 = time.time()
    print(f"simulated {nsteps} steps for a {n}-pendulum in {t1-t0:2.2f}s")

    # if use_angles:
    Ra = (thetas.clone().T)[:,:,None]
    Va = (dthetas.clone().T)[:,:,None]

    # Rscale = torch.sqrt(Ra.pow(2).mean())
    # Vscale = torch.sqrt(Va.pow(2).mean())
    # Ra /= Rscale
    # Va /= Vscale



    # else:
    x,y,vx,vy = get_coordinates_from_angle(thetas,dthetas)

    # animate_pendulum(x.numpy(), y.numpy(),vx.numpy(),vy.numpy())


    x = x.T
    y = y.T
    vx = vx.T
    vy = vy.T
    R = torch.cat((x[:,1:,None],y[:,1:,None]),dim=2)
    V = torch.cat((vx[:,1:,None],vy[:,1:,None]),dim=2)

    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    Rina, Routa = convert_snapshots_to_future_state_dataset(nskip, Ra)
    Vina, Vouta = convert_snapshots_to_future_state_dataset(nskip, Va)

    ndata = Rout.shape[0]
    print(f'Number of data: {ndata}')
    assert n_train+n_val <= ndata, f"The number of datasamples: {ndata} is less than the number of training samples: {n_train} + the number of validation samples: {n_val}"

    ndata_rand = 0 + np.arange(ndata)
    if shuffle:
        np.random.shuffle(ndata_rand)
    train_idx = ndata_rand[:n_train]
    val_idx = ndata_rand[n_train:n_train + n_val]

    Rin_train = Rin[train_idx]
    Rout_train = Rout[train_idx]
    Vin_train = Vin[train_idx]
    Vout_train = Vout[train_idx]

    Rina_train = Rina[train_idx]
    Routa_train = Routa[train_idx]
    Vina_train = Vina[train_idx]
    Vouta_train = Vouta[train_idx]

    if use_val:
        Rina_val = Rina[val_idx]
        Routa_val = Routa[val_idx]
        Vina_val = Vina[val_idx]
        Vouta_val = Vouta[val_idx]
        Rin_val = Rin[val_idx]
        Rout_val = Rout[val_idx]
        Vin_val = Vin[val_idx]
        Vout_val = Vout[val_idx]

    z = torch.arange(1,n+1)
    # z = torch.ones(n) #Should I go for this or torch arange(1,n+1)?

    z = z.to(device)
    M = M.to(device)
    Rin_train = Rin_train.to(device)
    Rout_train = Rout_train.to(device)
    Vin_train = Vin_train.to(device)
    Vout_train = Vout_train.to(device)
    Rina_train = Rina_train.to(device)
    Routa_train = Routa_train.to(device)
    Vina_train = Vina_train.to(device)
    Vouta_train = Vouta_train.to(device)

    Rin_val = Rin_val.to(device)
    Rout_val = Rout_val.to(device)
    Vin_val = Vin_val.to(device)
    Vout_val = Vout_val.to(device)
    Rina_val = Rina_val.to(device)
    Routa_val = Routa_val.to(device)
    Vina_val = Vina_val.to(device)
    Vouta_val = Vouta_val.to(device)


    dataset_train = DatasetFutureState(data_type,Rin_train, Rout_train, z, Vin_train, Vout_train, m=M,device=device,Rin2=Rina_train, Rout2=Routa_train, Vin2=Vina_train, Vout2=Vouta_train)
    if use_angles:
        dataset_train.useprimary = False
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if use_val:
        dataset_val = DatasetFutureState(data_type,Rin_val, Rout_val, z, Vin_val, Vout_val, m=M,device=device, Rin2=Rina_val, Rout2=Routa_val, Vin2=Vina_val, Vout2=Vouta_val)
        if use_angles:
            dataset_val.useprimary = False
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    return dataloader_train, dataloader_val




def load_pendulum_data(file,data_type,device,nskip,n_train,n_val,use_val,use_test,batch_size, shuffle=True):
    M = torch.tensor([2.0, 0.5])
    N = n_train+use_val *n_val + nskip
    def double_pendulum_system(x, L1=3.0, L2=2.0, m1=2.0, m2=0.5, g=9.8):
        # a system of differential equations defining a double pendulum
        # from http://www.myphysicslab.com/dbl_pendulum.html
        theta1 = x[0]
        theta2 = x[1]
        omega1 = x[2]
        omega2 = x[3]
        dtheta1 = omega1
        dtheta2 = omega2
        domega1 = (-g * (2 * m1 + m2) * torch.sin(theta1) -
                   m2 * g * torch.sin(theta1 - 2 * theta2) -
                   2 * torch.sin(theta1 - theta2) * m2 *
                   (omega2 ** 2 * L2 + omega1 ** 2 * L1 * torch.cos(theta1 - theta2))) / (L1 * (2 * m1 + m2 - m2 * torch.cos(2 * theta1 - 2 * theta2)))
        domega2 = (2 * torch.sin(theta1 - theta2) * (omega1 ** 2 * L1 * (m1 + m2) +
                                                     g * (m1 + m2) * torch.cos(theta1) + omega2 ** 2 * L2 * m2 *
                                                     torch.cos(theta1 - theta2))) / (L2 * (2 * m1 + m2 - m2 * torch.cos(2 * theta1 - 2 * theta2)));
        f = torch.tensor([dtheta1, dtheta2, domega1, domega2])

        return f

    def integrateSystem(x0, dt, N,m1,m2):
        x = x0
        X = torch.zeros(4, N + 1)
        X[:, 0] = x
        for i in range(N):
            k1 = double_pendulum_system(x,m1=m1,m2=m2)
            k2 = double_pendulum_system(x + dt / 2 * k1,m1=m1,m2=m2)
            k3 = double_pendulum_system(x + dt / 2 * k2,m1=m1,m2=m2)
            k4 = double_pendulum_system(x + dt * k3,m1=m1,m2=m2)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            X[:, i + 1] = x

        t = torch.linspace(0, N * dt, N + 1)

        return X, t

    def getCoordsVelFromAngles(x, L1=3.0, L2=2.0):
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]

        x = L1 * torch.sin(theta1)
        y = -L1 * torch.cos(theta1)
        r1 = torch.cat((x[:,None],y[:,None]),dim=1)

        x = x + L2 * torch.sin(theta2)
        y = y - L2 * torch.cos(theta2)
        r2 = torch.cat((x[:,None],y[:,None]),dim=1)

        vx = dtheta1 * L1 * torch.cos(theta1)
        vy = dtheta1 * L1 * torch.sin(theta1)
        v1 = torch.cat((vx[:,None],vy[:,None]),dim=1)

        vx = vx + dtheta2 * L2 * torch.cos(theta2)
        vy = vy + dtheta2 * L2 * torch.sin(theta2)
        v2 = torch.cat((vx[:,None],vy[:,None]),dim=1)

        # v2 = v1 + (dtheta2 * L2 * torch.cos(theta2), dtheta2 * L2 * torch.sin(theta2))
        # X = torch.cat((xm1.unsqueeze(0), ym1.unsqueeze(0), xm2.unsqueeze(0), ym2.unsqueeze(0)))
        return r1,r2,v1,v2

    def getAnglesFromCoords(x, L1=3.0, L2=2.0):
        xm1 = x[0]
        xm2 = x[1]
        theta1 = torch.arcsin(xm1 / L1)
        theta2 = torch.arcsin((xm2 - xm1) / L2)

        return torch.tensor([[theta1], [theta2]])

    theta0 = torch.tensor([3 * np.pi / 4, np.pi / 2, 0, 0])
    Theta, t = integrateSystem(theta0, 0.01, N, m1=M[0], m2=M[1])
    r1,r2,v1,v2 = getCoordsVelFromAngles(Theta)
    R = torch.cat((r1[:,None,:],r2[:,None,:]),dim=1)
    V = torch.cat((v1[:,None,:],v2[:,None,:]),dim=1)

    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)

    ndata = Rout.shape[0]
    print(f'Number of data: {ndata}')
    assert n_train+n_val <= ndata, f"The number of datasamples: {ndata} is less than the number of training samples: {n_train} + the number of validation samples: {n_val}"

    ndata_rand = 0 + np.arange(ndata)
    if shuffle:
        np.random.shuffle(ndata_rand)
    train_idx = ndata_rand[:n_train]
    val_idx = ndata_rand[n_train:n_train + n_val]
    test_idx = ndata_rand[n_train + n_val:]

    Rin_train = Rin[train_idx]
    Rout_train = Rout[train_idx]
    Vin_train = Vin[train_idx]
    Vout_train = Vout[train_idx]


    if use_val:
        Rin_val = Rin[val_idx]
        Rout_val = Rout[val_idx]
        Vin_val = Vin[val_idx]
        Vout_val = Vout[val_idx]

    z = torch.tensor([1,2])
    dataset_train = DatasetFutureState(data_type,Rin_train, Rout_train, z, Vin_train, Vout_train, m=M,device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if use_val:
        dataset_val = DatasetFutureState(data_type,Rin_val, Rout_val, z, Vin_val, Vout_val, m=M,device=device)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    return dataloader_train, dataloader_val


def load_MD_data(file,data_type,device,nskip,n_train,n_val,use_val,use_test,batch_size, shuffle=True, use_endstep=False):
    """
    a function for loading molecular dynamics data
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

    z = z[1::3]

    #We rescale the data
    Rscale = torch.sqrt(R.pow(2).mean())
    Vscale = torch.sqrt(V.pow(2).mean())
    R /= Rscale
    V /= Vscale
    # Rscale=1
    # Vscale=1

    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    Fin, Fout = convert_snapshots_to_future_state_dataset(nskip, F)
    KEin, KEout = convert_snapshots_to_future_state_dataset(nskip, KE)
    PEin, PEout = convert_snapshots_to_future_state_dataset(nskip, PE)

    R = None
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
    test_idx = ndata_rand[n_train + n_val:]
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
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        dataloader_val = None


    if use_test:
        dataset_test = DatasetFutureState(data_type,Rin_test, Rout_test, z, Vin_test, Vout_test, Fin_test, Fout_test, KEin_test, KEout_test, PEin_test, PEout_test, masses,device=device, rscale=Rscale, vscale=Vscale)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        dataloader_test = None

    if use_endstep:
        dataset_endstep = DatasetFutureState(data_type,Rin_endstep, Rout_endstep, z, Vin_endstep, Vout_endstep, m=masses,device=device, rscale=Rscale, vscale=Vscale,nskip=nskip)
        dataloader_endstep = DataLoader(dataset_endstep, batch_size=1, shuffle=False, drop_last=False)
    else:
        dataloader_endstep = None


    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep

def load_protein_data(file, data_type,device,n_train,batch_size, shuffle=False):
    """
    function for loading protein data

    """
    data = np.load(file,allow_pickle=True)

    seq = data['seq']
    rCa = data['rCa']
    rCb = data['rCb']
    rN = data['rN']
    pssm = data['pssm']
    entropy = data['entropy']
    log_units = data['log_units']
    ndata = len(seq)
    print('Number of datapoints={:}'.format(ndata))
    ndata_rand = 0 + np.arange(ndata)
    if n_train < 0:
        n_train = ndata
    if shuffle:
        np.random.shuffle(ndata_rand)
    train_idx = ndata_rand[:n_train]
    # val_idx = ndata_rand[n_train:n_train + n_val]
    # test_idx = ndata_rand[n_train + n_val:]

    collator = GraphCollate()
    dataset_train = Dataset_protein(data_type, seq[train_idx],rCa[train_idx],rCb[train_idx],rN[train_idx],pssm[train_idx],entropy[train_idx],device,log_units)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)

    return dataloader_train





class DatasetFutureState(data.Dataset):
    """
    A dataset type for future state predictions of molecular dynamics data
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




class GraphCollate:
    """
    A collating function for proteins, which enables the possibility of batchsizes larger than 1.
    """
    def __init__(self):
        return

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_collate(self, data):
        """
        This functions collates our data and transforms it into torch format. numpy arrays are padded according to the longest sequence in the batch in all dimensions.
        The padding mask is created according to mask_var and mask_dim, and is appended as the last variable in the output.
        args:
            data - Tuple of length nb, where nb is the batch size.
            data[0] contains the first batch and will also be a tuple with length nv, equal to the number of variables in a batch.
            data[0][0] contains the first variable of the first batch, this should also be a tuple with length nsm equal to the number of samples in the variable, like R1,R2,R3 inside coords.
            data[0][0][0] contains the actual data, and should be a numpy array.
            If any numpy array of ndim=0 is encountered it is assumed to be a string object, in which case it is turned into a string rather than a torch object.
            The datatype of the output is inferred from the input.
        return:
            A tuple of variables containing the input variables in order followed by the mask.
            Each variable is itself a tuple of samples
        """
        # find longest sequence
        nb = len(data)    # Number of batches
        nv = len(data[0]) # Number of variables in each batch

        seqs, coords,coords_init,Ms, pssms, entropy = zip(*data)
        batchs = [i+0*datai[0] for i,datai in enumerate(data)]
        batch = torch.cat(batchs)
        seq = torch.cat(seqs)
        coord = torch.cat(coords, dim=0)
        coord_init = torch.cat(coords_init,dim=0)
        edge_index = radius_graph(coord_init, 20.0, batch)
        edge_index_all = radius_graph(coord_init, 10000.0, batch,max_num_neighbors=99999)
        pssm = torch.cat(pssms, dim=0)
        entropy = torch.cat(entropy, dim=0)
        M = torch.cat(Ms)
        Msrc = M[edge_index_all[0]]
        Mdst = M[edge_index_all[1]]
        MM = Msrc * Mdst
        edge_index_all = [edge_index_all[0][MM], edge_index_all[1][MM]]
        return seq,batch, coord, M, pssm, entropy, edge_index,edge_index_all


def compute_all_connections(n,mask=None, include_self_connections=False, device='cpu'):
    """
    A simple function that computes all possible connections between particles, essentially a full edge graph
    """
    tmp = torch.arange(n, device=device)
    if mask is not None:
        tmp = tmp[mask]
    I = tmp.repeat_interleave(len(tmp))
    J = tmp.repeat(len(tmp))
    if not include_self_connections:
        m = I != J
        I = I[m]
        J = J[m]
    return I,J




class Dataset_protein(data.Dataset):
    """
    Dataset for proteins

    data_type:      The type of data, for now it is just proteins
    seq:            The amino sequence in numerical form (typically 0-19 are used for the most common amino acids.)
    rCa:            The 3D coordinates of Carbon alpha atom in each amino acid
    rCb:            The 3D coordinates of the carbon beta atom in each amino acid
    rN:             The 3D coordinates of teh Nitrogen atom in each amino acid
    device:         Device the data should be stored on
    log_units:      The logarithmic unit type of the 3D coordinates rCa,rCb,rN is in. (nanometers = -9, Angstrom= -10, m=0)
    pos_only:       Whether we only store positions or also velocities (for proteins this should always be true)
    internal_log_units: The logarithmic unit type we wish to do the calculations in (this is the unit type the constraints for instance are given in)
    """
    def __init__(self, data_type,seq, rCa,rCb,rN, pssm, entropy, device, log_units, pos_only=True,internal_log_units=-10):
        self.log_units = log_units
        self.internal_log_units = internal_log_units
        self.scale = 10**(log_units-internal_log_units)
        self.seq = [torch.from_numpy(seqi).to(device=device) for seqi in seq]
        self.rCa = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rCa]
        self.rCb = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rCb]
        self.rN = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rN]
        self.pssm = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype()) for tmp in pssm]
        self.entropy = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype()) for tmp in entropy]
        self.device = device
        self.rscale = 1 #The constraints are in Angstrom which is also what we have scaled to data to be in, so rscale should be unity.
        self.vscale = 1 #This is not used for proteins at all, since there are no velocity component
        self.data_type = data_type
        self.pos_only = pos_only
        return

    def __getitem__(self, index):
        seq = self.seq[index]
        rCa = self.rCa[index] #This gives coordinates in Angstrom, with a typical amino acid binding distance of 3.8 A
        rCb = self.rCb[index] #This gives coordinates in Angstrom, with a typical amino acid binding distance of 3.8 A
        rN = self.rN[index] #This gives coordinates in Angstrom, with a typical amino acid binding distance of 3.8 A
        pssm = self.pssm[index]
        entropy = self.entropy[index]
        m1 = rCa[:,0] != 0.0
        m2 = rCb[:,0] != 0.0
        m3 = rN[:,0] != 0.0
        M = m1 * m2 * m3

        coords = torch.cat([rCa,rCb,rN],dim=-1)
        rCa_init = torch.zeros_like(rCa)
        rCa_init[:,0] = torch.arange(rCa.shape[0])
        coords_init = torch.cat([rCa_init,rCa_init,rCa_init],dim=-1)

        return seq, coords,coords_init, M,pssm, entropy

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'