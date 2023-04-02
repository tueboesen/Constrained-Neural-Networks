import inspect
import math
import os
import time
from os.path import exists

import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_cluster import radius_graph

def data_split(features,metafile,n_train,n_val,n_test):
    if exists(metafile):
        with np.load(metafile) as mf:
            ndata = mf['ndata']
            assert len(features[list(features)[0]]) == ndata, "The number of data points in the dataset does not match the number in the metadata."
            idx_train = mf['idx_train']
            idx_val = mf['idx_val']
            idx_test = mf['idx_test']
    else:
        ndata = len(features[list(features)[0]])
        ndata_rand = 0 + np.arange(ndata)
        np.random.shuffle(ndata_rand)
        idx_train = ndata_rand[:n_train]
        idx_val = ndata_rand[n_train:n_train + n_val]
        idx_test = ndata_rand[n_train + n_val:n_train + n_val + n_test]
        os.makedirs(os.path.dirname(metafile), exist_ok=True)
        np.savez(metafile, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,ndata=ndata)

    f_train = {}
    f_val = {}
    f_test = {}
    for key, feature in features.items():
        f_train[key] = feature[idx_train]
        f_val[key] = feature[idx_val]
        f_test[key] = feature[idx_test]
    return f_train, f_val, f_test

def attach_edge_generator(dataloaders,edge_generator):
    for key, dataloader in dataloaders.items():
        dataloader.generate_edges = edge_generator
    return dataloaders



def generate_dataloaders(f_train,f_val,f_test,batchsize_train,batchsize_val,use_val,use_test,rscale=1,vscale=1,collate_vars_fnc=None):

    dataloaders = {}
    dataset_train = DatasetFutureState(rscale=rscale,vscale=vscale,**f_train)
    dataloader_train = Dataloader_ext(dataset_train, batch_size=batchsize_train, shuffle=True, drop_last=True)
    dataloaders['train'] = dataloader_train

    if use_val:
        dataset_val = DatasetFutureState(rscale=rscale,vscale=vscale,**f_val)
        dataloader_val = Dataloader_ext(dataset_val, batch_size=batchsize_val, shuffle=False, drop_last=False)
        dataloaders['val'] = dataloader_val


    if use_test:
        dataset_test = DatasetFutureState(rscale=rscale,vscale=vscale,**f_test)
        dataloader_test = Dataloader_ext(dataset_test, batch_size=batchsize_val, shuffle=False, drop_last=False)
        dataloaders['test'] = dataloader_test

    if collate_vars_fnc is not None:
        for key, val in dataloaders.items():
            dataloaders[key].collate_vars = collate_vars_fnc
    return dataloaders



class DatasetFutureState(data.Dataset):
    """
    A dataset type for future state predictions.
    """
    def __init__(self, Rin, Rout, Vin, Vout, particle_type, particle_mass, data_id=1, rscale=1, vscale=1):
        self.Rin = Rin
        self.Rout = Rout
        self.Vin = Vin
        self.Vout = Vout
        self.particle_type = particle_type
        self.particle_mass = particle_mass
        self.rscale = rscale
        self.vscale = vscale
        self.data_id = data_id
        return

    def __getitem__(self, index):
        Rin = self.Rin[index]
        Rout = self.Rout[index]
        Vin = self.Vin[index]
        Vout = self.Vout[index]
        particle_type = self.particle_type[index]
        particle_mass = self.particle_mass[index]
        data_id = self.data_id

        return Rin, Rout, Vin, Vout, particle_type, particle_mass

    def __len__(self):
        return len(self.Rin)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'

    def __str__(self):
        return f"{self.__class__.__name__}(len={len(self.Rin)})"

class Dataloader_ext(DataLoader):

    @classmethod
    def collate_vars(cls,Rin, Rout, Vin, Vout, z, m):
        """
        We expect the data to have the following structure:
        [nb,nm,np,nd]
        nb is the number of batches
        nm is the number of models/molecules (so a group of particles that are interconnected, but does not have constraints across)
        np, is the number of particles within a model or molecule
        nd is the number of dimensions for each of those particles (x,y,z,vx,vy,vz) or (x,y,vx,vy) for instance
        """


        nb, nm, np, nd = Rin.shape

        Rin_vec = Rin.reshape(-1, Rin.shape[-1])
        Rout_vec = Rout.reshape(-1, Rin.shape[-1])
        Vin_vec = Vin.reshape(-1, Rin.shape[-1])
        Vout_vec = Vout.reshape(-1, Rin.shape[-1])


        # Rin_vec = Rin.reshape(-1, Rin.shape[2]*Rin.shape[3])
        # Rout_vec = Rout.reshape(-1, Rin.shape[2]*Rin.shape[3])
        # Vin_vec = Vin.reshape(-1, Rin.shape[2]*Rin.shape[3])
        # Vout_vec = Vout.reshape(-1, Rin.shape[2]*Rin.shape[3])
        z_vec = z.reshape(-1, 1)

        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[2]).to(device=Rin.device)

        m_vec = m.view(-1, m.shape[-1])

        x_vec = torch.cat([Rin_vec, Vin_vec], dim=-1)
        xout_vec = torch.cat([Rout_vec, Vout_vec], dim=-1)
        weights = (1 / m)#.repeat(1,1,1,x_vec.shape[-1]//m.shape[-1])
        return batch, x_vec, z_vec, m_vec, weights, xout_vec

    @classmethod
    def generate_edges(cls,batch,x,max_radius):
        Rin_vec = x[...,...]
        edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=120)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        wstatic = None
        return edge_src, edge_dst, wstatic

