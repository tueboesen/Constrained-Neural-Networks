import inspect
import math
import os
import time
from os.path import exists

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch_cluster import radius_graph

from src.utils import atomic_masses, convert_snapshots_to_future_state_dataset
from pendulum.npendulum import NPendulum, get_coordinates_from_angle
import multibodypendulum as mbp


def load_data_wrapper(c):
    """
    Wrapper function that handles the different supported data_types.
    """
    if c.type == 'multibodypendulum':
        dataloaders = load_multibodypendulum_data(c) # Standard method for loading data
    else:
        raise NotImplementedError(f"The data type {c.type} has not been implemented yet")
    return dataloaders

def data_split(c,features):
    if exists(c.metafile):
        with np.load(c.metafile) as mf:
            ndata = mf['ndata']
            assert len(features[list(features)[0]]) == ndata, "The number of data points in the dataset does not match the number in the metadata."
            idx_train = mf['idx_train']
            idx_val = mf['idx_val']
            idx_test = mf['idx_test']
    else:
        ndata = len(features[list(features)[0]])
        ndata_rand = 0 + np.arange(ndata)
        np.random.shuffle(ndata_rand)
        idx_train = ndata_rand[:c.n_train]
        idx_val = ndata_rand[c.n_train:c.n_train + c.n_val]
        idx_test = ndata_rand[c.n_train + c.n_val:c.n_train + c.n_val + c.n_test]
        os.makedirs(os.path.dirname(c.metafile), exist_ok=True)
        np.savez(c.metafile, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,ndata=ndata)

    f_train = {}
    f_val = {}
    f_test = {}
    for key, feature in features.items():
        f_train[key] = feature[idx_train]
        f_val[key] = feature[idx_val]
        f_test[key] = feature[idx_test]
    return f_train, f_val, f_test


def load_multibodypendulum_data(c):
    """
    Creates a multibody pendulum dataset and loads it into standard pytorch dataloaders.
    #TODO we should save this as a feature artifact that can be loaded and applied directly to the raw data
    """

    features = feature_transform_multibodypendulum(c)
    f_train, f_val, f_test = data_split(c,features)
    dataloaders = generate_dataloaders(c,f_train,f_val,f_test)
    dataloaders = attach_custom_edge_generator(dataloaders)
    return dataloaders

def attach_custom_edge_generator(dataloaders):
    for key, dataloader in dataloaders.items():
        dataloader.generate_edges = multibodypendulum_edges
    return dataloaders

def generate_dataloaders(c,f_train,f_val,f_test):

    dataloaders = {}
    dataset_train = DatasetFutureState(**f_train)
    dataloader_train = Dataloader_ext(dataset_train, batch_size=c.batchsize, shuffle=True, drop_last=True)
    dataloaders['train'] = dataloader_train

    if c.use_val:
        dataset_val = DatasetFutureState(**f_val)
        dataloader_val = Dataloader_ext(dataset_val, batch_size=c.batchsize * 100, shuffle=False, drop_last=False)
        dataloaders['val'] = dataloader_val


    if c.use_test:
        dataset_test = DatasetFutureState(**f_test)
        dataloader_test = Dataloader_ext(dataset_test, batch_size=c.batchsize * 100, shuffle=False, drop_last=False)
        dataloaders['test'] = dataloader_test
    return dataloaders


def feature_transform_multibodypendulum(c):
    """
    TODO save this function as an artifact with the model for reproduceability
    """
    with np.load(c.file) as data:
        theta = data['theta']
        dtheta = data['dtheta']
    theta = torch.from_numpy(theta)
    dtheta = torch.from_numpy(dtheta)
    x,y,vx,vy = mbp.MultiBodyPendulum.get_coordinates_from_angles(theta,dtheta)

    R = torch.cat((x.T[:,:,None],y.T[:,:,None]),dim=2)
    V = torch.cat((vx.T[:,:,None],vy.T[:,:,None]),dim=2)
    nskip = c.nskip
    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    npenduls = Rin.shape[1]
    n = Rin.shape[0]
    particle_type = torch.arange(1,npenduls+1)[None,:].repeat(n,1)
    mass = torch.ones((n,npenduls))

    features = {}
    features['Rin'] = Rin.to(c.device)
    features['Rout'] = Rout.to(c.device)
    features['Vin'] = Vin.to(c.device)
    features['Vout'] = Vout.to(c.device)
    features['particle_type'] = particle_type.to(c.device)
    features['mass'] = mass.to(c.device)
    return features

class DatasetFutureState(data.Dataset):
    """
    A dataset type for future state predictions.
    """
    def __init__(self, Rin, Rout, Vin, Vout, particle_type,mass,rscale=1,vscale=1):
        self.Rin = Rin
        self.Rout = Rout
        self.Vin = Vin
        self.Vout = Vout
        self.particle_type = particle_type
        self.mass = mass
        self.rscale = rscale
        self.vscale = vscale
        return

    def __getitem__(self, index):
        Rin = self.Rin[index]
        Rout = self.Rout[index]
        Vin = self.Vin[index]
        Vout = self.Vout[index]
        z = self.particle_type[index]
        m = self.mass[index]

        return Rin, Rout, Vin, Vout, z,m

    def __len__(self):
        return len(self.Rin)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + ')'

    def __str__(self):
        return f"{self.__class__.__name__}(len={len(self.Rin)})"

class Dataloader_ext(DataLoader):

    @classmethod
    def collate_vars(cls,Rin, Rout, Vin, Vout, z, m):
        nb, natoms, ndim = Rin.shape

        Rin_vec = Rin.reshape(-1, Rin.shape[-1])
        Rout_vec = Rout.reshape(-1, Rout.shape[-1])
        Vin_vec = Vin.reshape(-1, Vin.shape[-1])
        Vout_vec = Vout.reshape(-1, Vout.shape[-1])
        z_vec = z.reshape(1, -1)

        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        m_vec = m.view(nb, natoms, -1)

        x = torch.cat([Rin_vec, Vin_vec], dim=-1)
        xout = torch.cat([Rout_vec, Vout_vec], dim=-1)
        weights = (1 / m).repeat_interleave(ndim // m.shape[-1], dim=-1).repeat(1, 1, x.shape[-1] // ndim)
        return batch, x, z_vec, m_vec, weights, xout

    @classmethod
    def generate_edges(cls,batch,x,max_radius):
        Rin_vec = x[...,...]
        edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=120)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        wstatic = None
        return edge_src, edge_dst, wstatic

def multibodypendulum_edges(batch,x,max_radius,npenduls=5):
    # Rin_vec = x[...,:x.shape[-1]//2]
    nb = (torch.max(batch) + 1).item()
    a = torch.tensor([0])
    b = torch.arange(1, npenduls - 1).repeat_interleave(2)
    c = torch.tensor([npenduls - 1])
    I = torch.cat((a, b, c))

    bb1 = torch.arange(1, npenduls)
    bb2 = torch.arange(npenduls - 1)
    J = torch.stack((bb1, bb2), dim=1).view(-1)

    shifts = torch.arange(nb).repeat_interleave(I.shape[0]) * npenduls

    II = I.repeat(nb)
    JJ = J.repeat(nb)

    edge_src = (JJ + shifts).to(device=batch.device)
    edge_dst = (II + shifts).to(device=batch.device)

    wstatic = torch.ones_like(edge_dst)
    return edge_src, edge_dst, wstatic