import math
import os
import time
from typing import Dict, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F

from preprocess.train_force_and_energy_predictor import generate_FE_network
from src.utils import atomic_masses


class VelocityVerletIntegrator(torch.nn.Module):
    def __init__(self,dt,force_predictor):
        super().__init__()
        self.dt = dt
        self.fp = force_predictor
        return

    def r_step(self,r,v,a):
        dt = self.dt
        r = r + v * dt + 0.5 * a * dt**2
        return r

    def v_step(self,v,a,a_new):
        dt = self.dt
        v = v + 0.5 * (a + a_new) * dt
        return v

    def forward(self,r,v,m,nsteps):
        fp = self.fp

        F = fp(r)
        a = F / m

        for i in range(nsteps):
            r = self.r_step(r,v,a)
            F_new = fp(r)
            a_new = F_new / m
            v = self.v_step(v,a,a_new)
            a = a_new
        return r,v


class VelocityVerletIntegrator(torch.nn.Module):
    def __init__(self,dt,force_predictor,z):
        super().__init__()
        self.dt = dt
        self.fp = force_predictor
        self.z = z
        return

    def r_step(self,r,v,a):
        dt = self.dt
        r = r + v * dt + 0.5 * a * dt**2
        return r

    def v_step(self,v,a,a_new):
        dt = self.dt
        v = v + 0.5 * (a + a_new) * dt
        return v

    def forward(self,r,v,m,a=None,F0=None,F1=None):
        fp = self.fp
        z = self.z[:,None]
        if a is None: # This is used on the first step
            if F0 is None:
                r.requires_grad_(True)
                E = fp(r,z)
                F = -grad(E, r, create_graph=True)[0].requires_grad_(True)
            else:
                F = F0
            a = F / m

        r_new = self.r_step(r,v,a)

        if F1 is None:
            r_new.requires_grad_(True)
            E = fp(r_new, z)
            F_new = -grad(E, r, create_graph=True)[0].requires_grad_(True)
        else:
            F_new = F1

        a_new = F_new / m
        v_new = self.v_step(v,a,a_new)
        return r_new,v_new,a_new


def MDdataloader(file, position_converter=1,velocity_converter=1,force_converter=1):
    data = np.load(file)
    R = torch.from_numpy(data['R'])*position_converter
    V = torch.from_numpy(data['V'])*velocity_converter
    F = torch.from_numpy(data['F'])*force_converter
    z = torch.from_numpy(data['z'])
    m = atomic_masses(z)
    return R,V,F,z,m




if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    bohr = 5.29177208590000E-11 #Bohr radius [m]
    hbar = 1.05457162825177E-34 #Planck constant/(2pi) [J*s]
    Eh = 4.3597447222071E-18 #Hatree energy [J]
    am = 1.66053906660E-27 #atomic mass [kg]
    au_T = hbar / Eh #[s^-1]

    #velocties are given in bohr / au_t , we want it in Angstrom/fs
    # vel_converter = bohr / au_T * 1e-5

    # Next we have forces which are given in Hartree energy / bohr, we want it in [au*Angstrom/fs^2]
    # force_converter = Eh / bohr / (am * 1e20)


    model_file = './../pretrained_networks/force_energy_model.pt'
    mddata_file = './../../../data/MD/argon/argon.npz'

    dt = 0.1 # fs
    nsteps = 100000

    Rt,Vt,Ft,z,m = MDdataloader(mddata_file)
    r = Rt[0]
    v = Vt[0]
    a = None

    force_predictor = generate_FE_network(natoms=r.shape[0])
    force_predictor.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    force_predictor.eval()
    # force_predictor = None
    VVI = VelocityVerletIntegrator(dt,force_predictor,z)

    for i in range(nsteps):

        r,v,a = VVI(r,v,m[:,None],a=a)
        # r,v,a = VVI(r,v,m[:,None],a=a,F0=Ft[i],F1=Ft[i+1])

        dr = torch.mean(torch.norm(Rt[i+1] - r,p=2,dim=1))
        dv = torch.mean(torch.norm(Vt[i+1] - v,p=2,dim=1))
        print(f"{i}, dr={dr:0.2e}, dv={dv:0.2e}")

