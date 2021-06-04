import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter

from src.utils import smooth_cutoff, smooth_cutoff2


class NodesToEdges(nn.Module):
    def __init__(self, n_vec_in):
        super().__init__()
        self.mix = MixVecVec(n_vec_in)

    def forward(self,xn,xe_src,xe_dst, W):
        xe_grad = W[:,None,:] * (xn[xe_src] - xn[xe_dst])
        xe_ave = W[:,None,:] * (xn[xe_src] + xn[xe_dst]) / 2
        xe = self.mix(xe_grad,xe_ave)
        return xe


class EdgesToNodes(nn.Module):
    def __init__(self, n_vec_in):
        super().__init__()
        # self.dim_in = dim_in
        # self.dim_out = dim_out
        # self.norm = 1 / math.sqrt(num_neighbours)
        self.mix = MixVecVec(n_vec_in)

    def forward(self, xe, xe_src, xe_dst, W):
        xn_1 = scatter(W[:,None,:] * xe,xe_dst,dim=0)# * self.norm
        xn_2 = scatter(W[:,None,:] * xe,xe_src,dim=0)# * self.norm
        xn_div = xn_1 - xn_2
        xn_ave = xn_1 + xn_2
        xn = self.mix(xn_div, xn_ave)
        return xn

class FullyConnectedNet(nn.Module):
    """
    A simple fully connected network, with an activation function at the end
    """
    def __init__(self,dimensions,activation_fnc):
        super(FullyConnectedNet, self).__init__()
        self.layers = nn.ModuleList()
        self.nlayers = len(dimensions)-1
        for i in range(self.nlayers):
            ll = nn.Linear(dimensions[i],dimensions[i+1])
            self.layers.append(ll)
        self.activation = activation_fnc
        return

    def forward(self,x):
        for i in range(self.nlayers):
            x = self.layers[i](x)
        x = self.activation(x)
        return x



class WeightLearner(nn.Module):
    def __init__(self):
        super(WeightLearner, self).__init__()
        stdv = 1
        # self.M = nn.Parameter(torch.tensor([1.0,0.1,0.01,0.001]) * stdv)
        self.M = nn.Parameter(torch.randn(2) * stdv)
        # self.norm = torch.nn.Softmax(dim=1)
        return

    def forward(self,x):
        eps = 1e-9
        x1 = x
        x2 = x*x
        # x1n = 1/(x+eps)
        # x2n = 1/(x2+eps)

        xcat = torch.cat([x1[:,None],x2[:,None]],dim=1)
        x = xcat @ self.M
        # xcat = self.norm(xcat)

        return x





class MixVecVec(nn.Module):
    def __init__(self, n_vec_in):
        super(MixVecVec, self).__init__()
        stdv = 1
        M1 = nn.Parameter(torch.randn(n_vec_in,n_vec_in) * stdv)
        M2 = nn.Parameter(torch.randn(n_vec_in,n_vec_in) * stdv)
        self.M1 = M1
        self.M2 = M2
        # self.w = nn.Parameter(torch.randn(1))
        return

    def forward(self, x1,x2):

        x1M = (x1 @ self.M1)
        x2M = (x2 @ self.M2)
        x3 = (x1M + x2M) /2.0
        return x3


class ActivationVec(nn.Module):
    def __init__(self):
        super(ActivationVec, self).__init__()
        return

    def forward(self, x):
        xnorm = x.norm(dim=1)
        x = x * torch.tanh(xnorm)[:,None,:]
        return x



class MixVecScalar(nn.Module):
    def __init__(self, n_vec_in,n_scalar_in):
        super(MixVecScalar, self).__init__()
        stdv = 1
        M = nn.Parameter(torch.randn(n_vec_in,n_scalar_in) * stdv)
        self.M = M
        self.w = nn.Parameter(torch.randn(1))
        return

    def forward(self, x1,x2):

        x3 = (x1 @ self.M)*x2[:,None,:]
        x3 = x3 @ self.M.t()

        node_angle = 0.1 * self.w
        w_org, w_res = node_angle.cos(), node_angle.sin()

        x = w_org * x1 + w_res * x3
        return x

class PropagationBlock(nn.Module):
    def __init__(self, n_vec_in, n_scalar_in):
        super().__init__()

        n_edges = 10
        n_nodes = 5
        # self.WeightLearner1 = WeightLearner()
        # self.WeightLearner2 = WeightLearner()
        self.fc1 = FullyConnectedNet(dimensions=[1,n_vec_in],activation_fnc=torch.nn.functional.silu)
        self.fc2 = FullyConnectedNet(dimensions=[1,n_vec_in],activation_fnc=torch.nn.functional.silu)

        self.MixVecScalar = MixVecScalar(n_vec_in,n_scalar_in)
        self.MixVecVec = MixVecVec(n_vec_in)

        self.nodes_to_edges = NodesToEdges(n_vec_in)
        self.edges_to_nodes = EdgesToNodes(n_vec_in)

        self.activation = ActivationVec()

        return

    def forward(self, xn, xn_attr, xe_attr, xe_src, xe_dst):
        eps = 1e-9

        xn = self.MixVecScalar(xn, xn_attr)
        # xn = xn / (xn.std(dim=1)[:,None] + eps)

        weight = self.fc1(xe_attr[:,None])
        xe = self.nodes_to_edges(xn, xe_src, xe_dst, weight)

        xe = self.MixVecVec(xe,xe)
        # xe = xe / (xe.std(dim=1)[:,None] + eps)

        weight = self.fc2(xe_attr[:,None])
        xn = self.edges_to_nodes(xe, xe_src, xe_dst, weight)

        xn = self.activation(xn)
        # xn = xn / (xn.std(dim=1)[:,None] + eps)

        return xn

class network_eq_simple(nn.Module):
    """
    This network is designed to predict the 3D coordinates of a set of particles.
    """
    def __init__(self, n_vec_in, n_vec_latent, nlayers, nmax_atom_types=10,atom_type_embed_dim=8, max_radius=5):
        super().__init__()

        self.nlayers = nlayers
        self.n_vec_in = n_vec_in
        self.n_vec_latent = n_vec_latent
        self.x_dim = 3 * n_vec_in
        self.y_dim = 3 * n_vec_latent

        w = torch.empty((self.n_vec_in, self.n_vec_latent))
        nn.init.xavier_normal_(w,gain=1/math.sqrt(self.n_vec_in)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        self.projection_matrix = nn.Parameter(w)
        self.node_attr_embedder = torch.nn.Embedding(nmax_atom_types,atom_type_embed_dim)
        self.max_radius = max_radius

        self.h = torch.nn.Parameter(torch.randn(nlayers)*1e-3)
        self.PropagationBlocks = nn.ModuleList()
        # self.angles = nn.ModuleList()

        for i in range(nlayers):
            block = PropagationBlock(n_vec_latent, atom_type_embed_dim)
            self.PropagationBlocks.append(block)
        #     angle = nn.Linear(xn_dim,1)
        #     self.angles.append(angle)
        return

    def make_matrix_semi_unitary(self, M,debug=False):
        I = torch.eye(M.shape[-2])
        if debug:
            M_org = M.clone()
        for i in range(10):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - M_org @ M_org.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        return M


    def project(self,y):
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        x = y @ M.t()
        return x

    def uplift(self,x):
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        y = x @ M
        return y

    def apply_constraints(self, y, n=1):
        for j in range(n):
            x = self.project(y)
            c = constraint(x.t(), d)
            lam = dConstraintT(c, x3.t())
            lam = self.uplift(lam.t())

            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam.norm()
                lsiter = 0
                while True:
                    xtry = x - alpha * lam
                    x3 = self.project(xtry)
                    ctry = constraint(x3.t(), d)
                    if torch.norm(ctry) < torch.norm(c):
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0:
                    alpha = alpha * 1.5
            x = x - alpha * lam
        return x

    def forward(self, x, node_attr, batch):
        #x should have the shape (nb,3,nv)

        node_attr_embedded = self.node_attr_embedder(node_attr.to(dtype=torch.int64)).squeeze()
        pos = x[:,:,-1]

        edge_index = radius_graph(pos, self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        y = self.uplift(x)
        # x2 = self.project(y)
        for i in range(self.nlayers):
            edge_vec = x[:,:,-1][edge_src] - x[:,:,-1][edge_dst]
            edge_len = edge_vec.norm(dim=1)
            w = smooth_cutoff(edge_len / self.max_radius) / edge_len

            y_org = y.clone() # Make sure this actually becomes a clone and not just a pointer
            y = self.PropagationBlocks[i](y, node_attr_embedded, w,edge_src,edge_dst)
            y = y_org + self.h[i] * y
            # y = self.apply_constraints(y, n=100)
            x = self.project(y)

        return x