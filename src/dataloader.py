import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data

from src.utils import atomic_masses, convert_snapshots_to_future_state_dataset



class DatasetFutureState(data.Dataset):
    def __init__(self, Rin, Rout, z, Vin, Vout, Fin=None, Fout=None, KEin=None, KEout=None, PEin=None, PEout=None, m=None, device='cpu', rscale=1, vscale=1):
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
        self.particles_pr_node = Rin.shape[-1] // 3
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
        Rin = self.Rin[index].to(device=device)
        Rout = self.Rout[index].to(device=device)
        z = self.z[:,None].to(device=device)
        Vin = self.Vin[index].to(device=device)
        Vout = self.Vout[index].to(device=device)
        if self.m is not None:
            m = self.m[:,None]
        else:
            m = 0
        if self.useF:
            Fin = self.Fin[index].to(device=device)
            Fout = self.Fout[index].to(device=device)
        else:
            Fin = 0
            Fout = 0
        if self.useKE:
            KEin = self.KEin[index].to(device=device)
            KEout = self.KEout[index].to(device=device)
        else:
            KEin = 0
            KEout = 0
        if self.usePE:
            PEin = self.PEin[index].to(device=device)
            PEout = self.PEout[index].to(device=device)
        else:
            PEin = 0
            PEout = 0
        return Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout,PEin, PEout,m

    def __len__(self):
        return len(self.Rin)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'


def load_data(file,device,nskip,n_train,n_val,use_val,use_test,batch_size, shuffle=True):
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

    #Analyse the system for creating constraints
    # n_particles = R.shape[-1]//3
    # if particles_pr_node != n_particles:
    #
    # Ro = R[:,0::3,:]
    # Rh1 = R[:,1::3,:]
    # Rh2 = R[:,2::3,:]
    #
    # Vo = V[:,0::3,:]
    # Vh1 = V[:,1::3,:]
    # Vh2 = V[:,2::3,:]
    #
    # Roh1 = Ro - Rh1
    # Roh2 = Ro - Rh2
    # Rh1h2 = Rh1 - Rh2
    # doh1 = Roh1.norm(dim=-1)
    # doh2 = Roh2.norm(dim=-1)
    # dh1h2 = Rh1h2.norm(dim=-1)
    #
    # doh1_mean = doh1.mean()
    # doh2_mean = doh2.mean()
    # dh1h2_mean = dh1h2.mean()
    #
    # doh = 0.9608
    # dhh = 1.5118

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

    Fin = None
    Fout = None
    KEin = None
    KEout = None
    PEin = None
    PEout = None
    # Fin_train = None
    # Fout_train = None
    # KEin_train = None
    # KEout_train = None
    # PEin_train = None
    # PEout_train = None

    dataset_train = DatasetFutureState(Rin_train, Rout_train, z, Vin_train, Vout_train, Fin_train, Fout_train, KEin_train, KEout_train, PEin_train, PEout_train, masses,device=device, rscale=Rscale, vscale=Vscale)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    if use_val:
        dataset_val = DatasetFutureState(Rin_val, Rout_val, z, Vin_val, Vout_val, Fin_val, Fout_val, KEin_val, KEout_val, PEin_val, PEout_val, masses,device=device, rscale=Rscale, vscale=Vscale)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        dataloader_val = None


    if use_test:
        dataset_test = DatasetFutureState(Rin_test, Rout_test, z, Vin_test, Vout_test, Fin_test, Fout_test, KEin_test, KEout_test, PEin_test, PEout_test, masses,device=device, rscale=Rscale, vscale=Vscale)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        dataloader_test = None

    return dataloader_train, dataloader_val, dataloader_test