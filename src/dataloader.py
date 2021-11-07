import inspect

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch_cluster import radius_graph

from src.utils import atomic_masses, convert_snapshots_to_future_state_dataset


def load_data(file,data_type,device,nskip,n_train,n_val,use_val,use_test,batch_size, shuffle=True, use_endstep=False):
    """
    Wrapper function that handles the different supported data_types.
    """
    if data_type == 'water':
        dataloader_train, dataloader_val, dataloader_test, dataloader_endstep =load_MD_data(file, data_type, device, nskip, n_train, n_val, use_val, use_test, batch_size, shuffle=shuffle, use_endstep=use_endstep)
    elif data_type == 'protein':
        dataloader_train, dataloader_val, dataloader_test = load_protein_data(file, data_type, device, n_train, n_val, use_val, use_test, batch_size, shuffle=shuffle)
        dataloader_endstep = None
    else:
        NotImplementedError("The data_type={:} has not been implemented in function {:}".format(data_type, inspect.currentframe().f_code.co_name))
    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep

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


    return dataloader_train, dataloader_val, dataloader_test, dataloader_endstep

def load_protein_data(file,data_type,device,n_train,n_val,use_val,use_test,batch_size, shuffle=False):
    """
    function for loading protein data

    """
    data = np.load(file,allow_pickle=True)

    seq = data['seq']
    rCa = data['rCa']
    rCb = data['rCb']
    rN = data['rN']
    ndata = len(seq)
    print('Number of datapoints={:}'.format(ndata))
    ndata_rand = 0 + np.arange(ndata)
    if shuffle:
        np.random.shuffle(ndata_rand)
    train_idx = ndata_rand[:n_train]
    val_idx = ndata_rand[n_train:n_train + n_val]
    test_idx = ndata_rand[n_train + n_val:]



    collator = GraphCollate()
    dataset_train = Dataset_protein(data_type, seq[train_idx],rCa[train_idx],rCb[train_idx],rN[train_idx],device=device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)

    if use_val:
        dataset_val = Dataset_protein(data_type, seq[val_idx], rCa[val_idx], rCb[val_idx], rN[val_idx], device=device)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collator)
    else:
        dataloader_val = None

    if use_test:
        dataset_test = Dataset_protein(data_type, seq[test_idx], rCa[test_idx], rCb[test_idx], rN[test_idx], device=device)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator)
    else:
        dataloader_test = None

    return dataloader_train, dataloader_val, dataloader_test





class DatasetFutureState(data.Dataset):
    """
    A dataset type for future state predictions of molecular dynamics data
    """
    def __init__(self, data_type, Rin, Rout, z, Vin, Vout, Fin=None, Fout=None, KEin=None, KEout=None, PEin=None, PEout=None, m=None, device='cpu', rscale=1, vscale=1, nskip=1, pos_only=False):
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

        seqs, coords,coords_init,Ms = zip(*data)
        batchs = [i+0*datai[0] for i,datai in enumerate(data)]
        batch = torch.cat(batchs)
        seq = torch.cat(seqs)
        coord = torch.cat(coords, dim=0)
        coord_init = torch.cat(coords_init,dim=0)
        edge_index = radius_graph(coord_init, 20.0, batch)
        edge_index_all = radius_graph(coord_init, 10000.0, batch,max_num_neighbors=99999)
        M = torch.cat(Ms)
        Msrc = M[edge_index_all[0]]
        Mdst = M[edge_index_all[1]]
        MM = Msrc * Mdst
        edge_index_all = [edge_index_all[0][MM], edge_index_all[1][MM]]
        return seq,batch, coord, M, edge_index,edge_index_all


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
    """
    def __init__(self, data_type,seq, rCa,rCb,rN, device, pos_only=True):
        self.scale = 1e1 #My data is saved in nanometers but we want it in Angstrom  #TODO Switch to logunits

        self.seq = [torch.from_numpy(seqi).to(device=device) for seqi in seq]
        self.rCa = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rCa]
        self.rCb = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rCb]
        self.rN = [torch.from_numpy(tmp).to(device=device,dtype=torch.get_default_dtype())*self.scale for tmp in rN]
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
        m1 = rCa[:,0] != 0.0
        m2 = rCb[:,0] != 0.0
        m3 = rN[:,0] != 0.0
        M = m1 * m2 * m3

        coords = torch.cat([rCa,rCb,rN],dim=-1)
        rCa_init = torch.zeros_like(rCa)
        rCa_init[:,0] = torch.arange(rCa.shape[0])
        coords_init = torch.cat([rCa_init,rCa_init,rCa_init],dim=-1)

        return seq, coords,coords_init, M

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'