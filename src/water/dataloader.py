import numpy as np
import torch

from src.dataloader_utils import data_split, generate_dataloaders
from src.utils import convert_snapshots_to_future_state_dataset, atomic_masses


def load_water_data(file,use_val,use_test,metafile,n_train,n_val,n_test,batchsize_train,batchsize_val,nskip,device,data_id):
    features, rscale, vscale = feature_transform_water(file,nskip,device)
    f_train, f_val, f_test = data_split(features,metafile,n_train,n_val,n_test)
    dataloaders = generate_dataloaders(f_train,f_val,f_test,batchsize_train,batchsize_val,use_val,use_test, rscale, vscale,collate_vars_fnc=collate_vars_water)
    return dataloaders





def feature_transform_water(file,nskip,device):
    """
    TODO save this function as an artifact with the models for reproduceability
    """
    with np.load(file) as data:
        R1 = torch.from_numpy(data['R1']).to(device=device,dtype=torch.get_default_dtype())
        R2 = torch.from_numpy(data['R2']).to(device=device,dtype=torch.get_default_dtype())
        R3 = torch.from_numpy(data['R3']).to(device=device,dtype=torch.get_default_dtype())
        V1 = torch.from_numpy(data['V1']).to(device=device,dtype=torch.get_default_dtype())
        V2 = torch.from_numpy(data['V2']).to(device=device,dtype=torch.get_default_dtype())
        V3 = torch.from_numpy(data['V3']).to(device=device,dtype=torch.get_default_dtype())
        particle_type = torch.from_numpy(data['z']).to(device=device)

    R = torch.cat([R1[:,:,None,:], R2[:,:,None,:], R3[:,:,None,:]], dim=2)
    V = torch.cat([V1[:,:,None,:], V2[:,:,None,:], V3[:,:,None,:]], dim=2)


    particle_mass = atomic_masses(particle_type).view(-1,3)
    particle_type = particle_type.view(-1, 3)

    rscale = torch.sqrt(R.pow(2).mean()).item()
    vscale = torch.sqrt(V.pow(2).mean()).item()
    R /= rscale
    V /= vscale

    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)

    particle_mass = particle_mass[...,None].repeat(Rin.shape[0],1,1,Rin.shape[-1]*2)
    particle_type = particle_type[...,None].repeat(Rin.shape[0],1,1,Rin.shape[-1]*2)

    features = {}
    features['Rin'] = Rin.to(device)
    features['Rout'] = Rout.to(device)
    features['Vin'] = Vin.to(device)
    features['Vout'] = Vout.to(device)
    features['particle_type'] = particle_type.to(device)
    features['particle_mass'] = particle_mass.to(device)
    return features, rscale, vscale


def collate_vars_water(Rin, Rout, Vin, Vout, z, m):
    """
    We expect the data to have the following structure:
    [nb,nm,np,nd]
    nb is the number of batches
    nm is the number of models/molecules (so a group of particles that are interconnected, but does not have constraints across)
    np, is the number of particles within a model or molecule
    nd is the number of dimensions for each of those particles (x,y,z,vx,vy,vz) or (x,y,vx,vy) for instance
    """


    nb, nm, np, nd = Rin.shape

    Rin_vec = Rin.reshape(-1, Rin.shape[2]*Rin.shape[3])
    Rout_vec = Rout.reshape(-1, Rin.shape[2]*Rin.shape[3])
    Vin_vec = Vin.reshape(-1, Rin.shape[2]*Rin.shape[3])
    Vout_vec = Vout.reshape(-1, Rin.shape[2]*Rin.shape[3])
    z_vec = z[:,:,:,0].reshape(-1, z.shape[-2])

    batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

    m_vec = m.view(-1, m.shape[-1])

    x_vec = torch.cat([Rin_vec, Vin_vec], dim=-1)
    xout_vec = torch.cat([Rout_vec, Vout_vec], dim=-1)
    weights = (1 / m)#.repeat(1,1,1,x_vec.shape[-1]//m.shape[-1])
    return batch, x_vec, z_vec, m_vec, weights, xout_vec
