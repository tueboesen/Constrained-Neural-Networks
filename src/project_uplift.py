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
        self.irreps_low = irreps_low
        self.irreps_high = irreps_high
        self.n_vec_low = ExtractIr(irreps_low, '1o').irreps_out.num_irreps
        self.n_vec_high = ExtractIr(irreps_high, '1o').irreps_out.num_irreps

        w = torch.empty((self.n_vec_low, self.n_vec_high))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.n_vec_low)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.M = torch.nn.Parameter(w)
        # self.register_buffer("M", w)
        # self.make_matrix_semi_unitary()

        M = w
        I = torch.eye(M.shape[-2],device=M.device)
        # errors = []
        for i in range(1000):
            M = M - 0.5 * (M @ M.t() - I) @ M
            # post_error = torch.norm(I - M @ M.t())
            # errors.append(post_error)

        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # mpl.use('TkAgg')
        # plt.figure()
        # plt.semilogy(errors)
        # plt.show()
        # plt.pause(1)
        self.register_buffer("Mu", M)
        # post_error = torch.norm(I - M @ M.t())
        # print(f"Deviation from unitary: {post_error:2.2e}")
        return

    def make_matrix_semi_unitary(self, debug=False):
        """
        Takes a matrix and makes it unitary.
        Originally this was intended to be used such that we could have trainable uplifting/projection operations, but it turns out that the semi_unitary operation is not stable enough for that.
        So instead we just define the inital projection / uplifting matrix, and then leave it as is, which seems to work.
        """
        M = self.M.clone()
        I = torch.eye(M.shape[-2],device=M.device)
        for i in range(100):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - self.M @ self.M.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        self.Mu = M
        return

    def uplift(self,x):
        """
        Takes a low dimensional variable and transforms it into a high dimensional variable
        """
        irreps_in = self.irreps_low
        irreps_out = self.irreps_high
        nd_out = irreps_out.dim
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                x_vec = x[:,idx0:idx1]
                break
            idx0 = idx1
        x2 = x_vec.reshape(x_vec.shape[0], -1, 3).transpose(1, 2)
        y2 = x2 @ self.Mu
        y_vec = y2.transpose(1,2).reshape(x.shape[0],-1)
        y = torch.zeros((x_vec.shape[0],nd_out),dtype=x.dtype,device=x.device)
        idx0=0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y[:,idx0:idx1] = y_vec
                break
            idx0 = idx1
        return y

    def project(self,y):
        """
        Takes a high dimensional space and projects it into a lower dimensional space
        """
        irreps_in = self.irreps_high
        irreps_out = self.irreps_low
        ir_vec = ExtractIr(irreps_in,'1o')
        idx0 = 0
        for mul, ir in irreps_in:
            li = 2 * ir.l + 1
            idx1 = idx0 + li*mul
            if ir == o3.Irrep('1o'):
                y_vec = y[:,idx0:idx1]
                break
            idx0 = idx1
        y2 = y_vec.reshape(y.shape[0],-1,3).transpose(1,2)
        x2 = y2 @ self.Mu.t()
        x = x2.transpose(1,2).reshape(y.shape[0],-1)
        return x


class ProjectUplift(torch.nn.Module):
    '''
    A projection and uplifting class for standard networks.
    '''
    def __init__(self,low_dim,high_dim):
        super(ProjectUplift, self).__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim
        w = torch.empty((self.low_dim, self.high_dim))
        torch.nn.init.xavier_normal_(w,gain=1/math.sqrt(self.low_dim)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        # self.M = torch.nn.Parameter(w)
        M = w
        I = torch.eye(M.shape[-2],device=M.device)
        for i in range(100):
            M = M - 0.5 * (M @ M.t() - I) @ M
        self.register_buffer("Mu", M)
        return

    def make_matrix_semi_unitary(self, debug=False):
        M = self.M.clone()
        I = torch.eye(M.shape[-2],device=M.device)
        for i in range(100):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - self.M @ self.M.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        self.Mu = M
        return

    def project(self, y):
        """
        Projects the high dimensional variable y, into the low dimensional space x by the multiplication of a semi-unitary matrix
        """
        x = y @ self.Mu.t()
        return x

    def uplift(self, x):
        """
        Uplifts the low dimensional variable x, into the high dimensional space y by the multiplication of a semi-unitary matrix
        """
        y = x @ self.Mu
        return y
