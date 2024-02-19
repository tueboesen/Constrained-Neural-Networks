import torch.nn.functional as F
import math
import time

import torch
import torch.nn as nn


class neural_network_resnet(nn.Module):
    """
    This network is designed to denoise images.
    """

    def __init__(self, dim_in, dim_latent, layers, min_con_fnc=None, con_type=None, discretization_method='leapfrog', penalty_strength=0, regularization_strength=0, orthogonal_projection=True):
        super().__init__()
        """
        node_dimn_latent:   The dimension of the latent space
        nlayers:            The number of propagationBlocks to include in the network 
        nmax_atom_types:    Max number of particle types to handle, for proteins this should be the types of amino acids
        atom_type_embed_dim: The out dimension of the embedding done to the atom_types
        max_radius:         The maximum radius used when building the edges in the graph between the particles
        con_fnc:            A handler to the constrain function wrapped in a nn.sequential.
        con_type:           The type of constraints being used (high, low, reg)
        dim:                The dimension that the data lives in (default 3)
        """
        self.nlayers = layers
        self.penalty_strength = penalty_strength
        self.regularization_strength = regularization_strength
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.orthogonal_K = orthogonal_projection
        self.lin = torch.nn.Linear(self.dim_in, self.dim_latent)
        # self.h = torch.nn.Parameter(torch.ones(layers) * 1e-2)
        if self.orthogonal_K:
            self.ortho = torch.nn.utils.parametrizations.orthogonal(self.lin)
        else:
            pass

        self.min_con_fnc = min_con_fnc
        self.con_fnc = min_con_fnc.c
        self.con_type = con_type
        self.discretization_method = discretization_method
        assert self.discretization_method in ['leapfrog', 'euler', 'rk4']

        # self.Open = nn.Conv2d(dim_in, dim_latent, kernel_size=3, padding=1) #TODO THIS NEEDS TO HAVE A REVERSE OPERATION, so we use K instead
        Convs1 = nn.ParameterList()
        Convs2 = nn.ParameterList()
        LN = nn.ParameterList()

        for i in range(layers):
            Ci = nn.Conv2d(dim_latent, dim_latent, kernel_size=5, padding=2)
            Convs1.append(Ci)
            Ci = nn.Conv2d(dim_latent, dim_latent, kernel_size=5, padding=2)
            Convs2.append(Ci)
            LNi = nn.InstanceNorm2d(dim_latent)
            LN.append(LNi)

        self.Convs1 = Convs1
        self.Convs2 = Convs2
        self.LN = LN
        self.h = 0.1
        # self.Close = nn.Conv2d(dim_latent, dim_in, kernel_size=5, padding=2)

        # self.params = nn.ModuleDict({
        #     "base": self.Convs1 + self.Convs2,
        #     "h": nn.ParameterList([self.h]),
        #     "close": nn.ModuleList([self.lin])
        # })
        return

    def uplift(self, x):
        x = torch.moveaxis(x, 1, -1)
        y = x @ self.lin.weight.T
        y = torch.moveaxis(y, -1, 1)
        return y

    @property
    def K(self):
        return self.lin.weight.T

    def project(self, y):
        y = torch.moveaxis(y, 1, -1)
        x = y @ self.lin.weight
        x = torch.moveaxis(x, -1, 1)
        return x

    def propagate_layer(self,i,y):
        dy = self.Convs1[i](y)
        dy = self.LN[i](dy)
        dy = F.silu(dy)
        y_new = self.Convs2[i](dy)
        return y_new

    def forward(self, x):
        if x.isnan().any():
            raise ValueError("NaN detected")

        h = self.h
        y = self.uplift(x)
        y_old = y
        reg = torch.tensor(0.0)

        for i in range(self.nlayers):

            y_new = self.propagate_layer(i,y)

            if self.penalty_strength > 0:
                if self.discretization_method == 'rk4':
                    q1 = self.con_fnc.constraint_penalty(y, self.project, self.uplift)
                    q2 = self.con_fnc.constraint_penalty(y + q1 / 2 * h, self.project, self.uplift)
                    q3 = self.con_fnc.constraint_penalty(y + q2 / 2 * h, self.project, self.uplift)
                    q4 = self.con_fnc.constraint_penalty(y + q3 * h, self.project, self.uplift)
                    dy = (q1 + 2 * q2 + 2 * q3 + q4) / 6
                else:
                    dy = self.con_fnc.constraint_penalty(y, self.project, self.uplift)
            else:
                dy = 0
            if self.discretization_method == 'leapfrog':
                tmp = y.clone()
                y = 2 * y - y_old - h * (y_new + self.penalty_strength * dy)
                y_old = tmp
            elif self.discretization_method == 'euler':
                y = y + h * (y_new - self.penalty_strength * dy)
            elif self.discretization_method == 'rk4':
                k1 = self.propagate_layer(i,y)
                k2 = self.propagate_layer(i,y + k1 * h/2)
                k3 = self.propagate_layer(i,y + k2 * h/2)
                k4 = self.propagate_layer(i,y + k3 * h)
                y_new = (k1 + 2 * k2 + 2 * k3 + k4) / 6
                y = y + h * (y_new - self.penalty_strength * dy)
            else:
                raise NotImplementedError(f"Discretization method {self.discretization_method} not implemented.")
            if self.min_con_fnc is not None and self.con_type == 'high':
                cv, _, _ = self.con_fnc.constraint_violation(y, project_fnc=self.project)
                y = self.min_con_fnc(y, project_fnc=self.project, uplift_fnc=self.uplift)
                reg = reg + (cv * cv).mean()
        x = self.project(y)
        if self.min_con_fnc is not None:
            cv, cv_mean, cv_max = self.con_fnc.constraint_violation(x)
            if reg == 0:
                reg = (cv * cv).mean()
            if self.con_type == 'low':
                x = self.min_con_fnc(x)
            cv, cv_mean, cv_max = self.con_fnc.constraint_violation(x) #We need this twice here since we want the regularization to be calculated before the projection is applied, but we want the actual constraint violation after the projection as well.
        else:
            cv_mean, cv_max = torch.tensor(-1.0), torch.tensor(-1.0)
        return x, cv_mean, cv_max, reg * self.regularization_strength
