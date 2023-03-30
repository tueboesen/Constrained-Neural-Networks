import math

import matplotlib.pyplot as plt
import torch
from e3nn import o3
from e3nn.nn import ExtractIr


class ProjectUpliftEQ(torch.nn.Module):
    '''
    A projection and uplifting class for equivariant variables.
    This is similar to the much simpler projectionUplifting class, but because of the equivariant data there is a lot more bookeeping to handle.
    Note that this will only work if the low dimension is purely a vector space for now.
    '''
    def __init__(self,irreps_low,irreps_high):
        super(ProjectUpliftEQ, self).__init__()
        irreps_low = o3.Irreps(irreps_low)
        irreps_high = o3.Irreps(irreps_high)
        self.irreps_low = irreps_low
        self.irreps_high = irreps_high
        self.n_vec_low = ExtractIr(irreps_low, '1o').irreps_out.num_irreps
        self.n_vec_high = ExtractIr(irreps_high, '1o').irreps_out.num_irreps

        self.lin = torch.nn.Linear(self.n_vec_low,self.n_vec_high)
        self.ortho = torch.nn.utils.parametrizations.orthogonal(self.lin)


        # w = torch.empty((self.n_vec_high, self.n_vec_low))
        # torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.n_vec_low)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.K = torch.nn.Parameter(w)
        return

    def extract_reps(self,x,irreps_in,irreps_extract=o3.Irrep('1o')):
        """
        Takes an irreps and extract a part of it
        """
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li * mul
            if ir == irreps_extract:
                x_extract = x[:, idx0:idx1]
                break
            idx0 = idx1
        return x_extract

    def insert_reps(self,x,irreps_out,irreps_in=o3.Irrep('1o')):
        """
        Takes an irrep and inserts it into a larger irreps.
        """
        nd_out = irreps_out.position_indices
        y = torch.zeros((x.shape[0],nd_out),dtype=x.dtype,device=x.device)
        idx0=0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == irreps_in:
                y[:,idx0:idx1] = x
                break
            idx0 = idx1
        return y

    def insert_reps_fast(self,x,nd=180,idx0=60,idx1=120):
        y = torch.zeros((x.shape[0],nd),dtype=x.dtype,device=x.device)
        y[:,idx0:idx1] = x
        return y

    def extract_reps_fast(self,x,idx0=60,idx1=120):
        x_extract = x[:,idx0:idx1]
        return x_extract

    def uplift(self,x):
        """
        Takes a low dimensional variable and transforms it into a high dimensional variable
        """
        ndims = x.shape
        x = x.view(-1,ndims[-1])
        irreps_in = self.irreps_low
        irreps_out = self.irreps_high
        x_vec = self.extract_reps(x, irreps_in)
        x2 = x_vec.view(x_vec.shape[0], -1, 3).transpose(1, 2)
        y2 = x2 @ self.lin.weight.T
        y_vec = y2.transpose(1,2).reshape(x.shape[0],-1)
        y = self.insert_reps(y_vec,irreps_out)
        if len(ndims) == 3:
            y = y.view(ndims[0],ndims[1],-1)
        return y

    def project(self,y):
        """
        Takes a high dimensional space and projects it into a lower dimensional space
        """
        ndims = y.shape
        y = y.view(-1,ndims[-1])
        irreps_in = self.irreps_high
        y_vec = self.extract_reps(y, irreps_in)
        y2 = y_vec.view(y.shape[0],-1,3).transpose(1,2)
        x2 = y2 @ self.lin.weight
        x = x2.transpose(1,2).reshape(y.shape[0],-1)
        if len(ndims) == 3:
            x = x.view(ndims[0],ndims[1],-1)
        return x