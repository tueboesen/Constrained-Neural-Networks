import inspect
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import grad

from src.vizualization import plot_water, plot_pendulum_snapshot_custom


def load_constraints(con,con_type,con_variables=None,rscale=1,vscale=1,device='cpu',debug_folder=None):
    """
    This is a wrapper function for loading constraints.
    """
    if con == 'water':
        r0 = con_variables['r0']
        r1 = con_variables['r1']
        r2 = con_variables['r2']
        l = torch.tensor([r0/rscale,r1/rscale,r2/rscale],device=device)
        con_fnc = Water(l,tol=1e-6/rscale,niter=100)
    elif con == 'n-pendulum':
        if con_type == 'stabhigh':
            niter = 1
        else:
            niter = 200
        converged_acc = 1e-4
        L = torch.tensor(con_variables['L'][0])
        con_fnc = MultiPendulum(L,niter=niter,tol=converged_acc,debug_folder=debug_folder)
    elif con == '':
        con_fnc = None
    else:
        raise NotImplementedError("The constraint chosen has not been implemented.")
    return con_fnc

def load_constraint_parameters(con,con_type,data_type,con_data='',model_specific=None):
    """
    Loads the constraints parameters, either from a data file if one is provided, or from the function constraint_hyperparameters.
    """
    if con_data == '':
        cv = constraint_hyperparameters(con,con_type,data_type,model_specific)
    else:
        cv = torch.load(con_data)
    return cv

def constraint_hyperparameters(con,con_type,data_type,model_specific):
    """
    Here we store some of the simpler constraint variables needed. More complicated constraints should be saved to a file and loaded instead.
    """
    cv = {}
    if con == 'chain':
        if data_type == 'proteins':
            cv['d0'] = 3.8 #Angstrom, ensure that your units are correct.
        else:
            NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con,data_type,inspect.currentframe().f_code.co_name))
    elif con == 'water':
        if data_type == 'water':
            cv['r0'] = 0.957
            cv['r1'] = 1.513
            cv['r2'] = 0.957
        else:
            NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con,data_type,inspect.currentframe().f_code.co_name))
    elif con == 'pendulum':
        cv['L1'] = 3.0
        cv['L2'] = 2.0
    elif con == 'n-pendulum' or con == 'n-pendulum-seq' or con == 'n-pendulum-seq-start':
        cv['L'] = model_specific['L']
    else:
        NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con, data_type, inspect.currentframe().f_code.co_name))
    return cv

class ConstraintTemplate(nn.Module):
    """
    This is the template class for all constraints.
    Each constraint should have this class as their parent.

    When making a constraint class, you only need to define an appropriate constraint function.

    If you supply high dimensional data y, the data needs to be supplied with a torch tensor K, such that x = y @ K and y = x @ K.T
    If you do not supply K, then K will essentially be an identity matrix, such that x=y

    Input:
        tol: the error tolerance for early stopping of the gradient descent operation.
        niter: Maximum number of gradient descent steps taken.
    """

    def __init__(self, tol,niter,sanity_check_upon_first_run=True,debug_folder=None):
        super(ConstraintTemplate, self).__init__()
        self.tol = tol
        self.n = niter
        self.sanity_check = sanity_check_upon_first_run
        self.debug_folder=debug_folder
        return

    def debug(self,x):
        raise NotImplementedError(
            "Debug function has not been implemented for {:}".format(self._get_name()))


    def constraint(self,x):
        raise NotImplementedError(
            "Constraint function has not been implemented for {:}".format(self._get_name()))

    def jacobian_transpose_times_constraint(self,x,c):
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
        return JTc

    def jacobian_transpose_times_constraint_backup(self,x,c):
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
        return JTc

    def constrain_stabilization(self, y,project=nn.Identity(),uplift=nn.Identity(),weight=1):
        """
        Calculates constraint stabilization as dy = J^T c
        """
        x = project(y)
        c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
        dx = self.jacobian_transpose_times_constraint(x,c)
        dx = weight * dx
        dy = uplift(dx)
        return dy


    def gradient_descent(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug=False):
        y_org = y.clone()
        for j in range(self.n):
            x = project(y)
            c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
            if j == 0:
                reg = c_error_mean
            if c_error_max < self.tol:
                break
            if debug:
                self.debug(x, c, extra=j)
            dx = self.jacobian_transpose_times_constraint(x,c)
            dx = weight * dx
            dy = uplift(dx)
            if j == 0:
                reg = c_error_mean
                alpha = 1.0 / dy.norm(dim=-1).mean()
            lsiter = 0
            while True:
                y_try = y - alpha * dy
                x_try = project(y_try)
                # y_try2 = uplift(x_try)
                # x_try2 = project(y_try2)
                c_try, c_error_mean_try, c_error_max_try = self.compute_constraint_violation(x_try)
                if c_error_max_try < c_error_max:
                    break
                alpha = alpha / 2
                lsiter = lsiter + 1
                if alpha == 0:
                    if c_error_max > 1e-2 and debug is False:
                        self.gradient_descent(y_org, project, uplift, weight, debug=True)
                    return y
            if lsiter == 0 and c_error_max_try > self.tol:
                alpha = alpha * 1.5
            if c_error_max_try < c_error_max:
                y = y - alpha * dy
        if j+1 >= self.n:
            print("problems detected!")
            # self.gradient_descent(y_org, project, uplift, weight, extra='n_exceeded', debug=True)
        # print(j)
        return y, reg


    def gradient_descent_batch(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug_idx=None):
        y_org = y.clone()
        nb = y.shape[0]
        alpha = torch.ones(nb,device=y.device)
        j = 0
        while True:
        # for j in range(self.n):
            idx_all = torch.arange(nb)
            x = project(y)
            c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
            if j == 0:
                reg = c_error_mean
            cm = c.abs().max(dim=1)[0]
            M = cm > self.tol
            idx = idx_all[M]
            if len(idx) == 0:
                break
            if debug_idx is not None:
                self.debug(x, c, extra=j, idx=debug_idx)
            dx = self.jacobian_transpose_times_constraint(x[idx],c[idx])
            dx = weight[idx] * dx
            dy = uplift(dx)
            lsiter = torch.zeros(len(idx))
            while True:
                y_try = y[idx] - alpha[idx,None,None] * dy
                x_try = project(y_try)
                c_try, c_error_mean_try, c_error_max = self.compute_constraint_violation(x_try)
                cm_try = c_try.abs().max(dim=1)[0]
                M_try = cm_try <= cm[idx]
                if M_try.all():
                    break
                idx_sel = idx[~M_try]
                alpha[idx_sel] = alpha[idx_sel] / 2
                lsiter[~M_try] = lsiter[~M_try] + 1
                if lsiter.max() > 100:
                    break
            M_increase = lsiter == 0
            idx_sel = idx[M_increase]
            alpha[idx_sel] = alpha[idx_sel] * 1.5
            ysel = y[idx] - alpha[idx,None,None] * dy
            yall = []
            count = 0
            for i in range(nb):
                if M[i]:
                    yall.append(ysel[count])
                    count = count + 1
                else:
                    yall.append(y[i])
            y = torch.stack(yall,dim=0)
            j += 1
            if j > self.n:
                # print(f"{cm.max()}")
                # if debug_idx is None:
                #     self.gradient_descent_batch(y_org,project,uplift,weight,debug_idx=cm.argmax())
                # x = project(y)
                # x_org = project(y_org)
                # c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
                # c_org, c_org_error_mean, c_org_error_max = self.compute_constraint_violation(x_org)
                # dx = self.jacobian_transpose_times_constraint(x, c)
                # dx2 = self.jacobian_transpose_times_constraint_backup(x, c)
                # self.debug(x,c)
                # self.debug(x_org,c_org,extra='org')
                # assert torch.allclose(dx,dx2,rtol=1e-7), "The jacobian is wrong!"

                break
        return y, reg

    def compute_constraint_violation(self, x):
        """
        Computes the constraint violation.
        """
        c = self.constraint(x)
        cabs = torch.abs(c)
        c_error_mean = torch.mean(cabs)
        c_error_max = torch.max(torch.abs(cabs))
        return c,c_error_mean, c_error_max

    def forward(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1, use_batch=True):
        if use_batch:
            y, reg = self.gradient_descent_batch(y,project,uplift,weight)
        else:
            nb = y.shape[0]
            y_new = []
            reg = 0
            for i in range(nb):
                tmp, regi = self.gradient_descent(y[i:i+1], project, uplift, weight[i:i+1])
                y_new.append(tmp)
                reg = reg + regi
            y = torch.cat(y_new, dim=0)
        return y, reg

class MultiPendulum(ConstraintTemplate):
    """
    This constraint function applies multi-body pendulum constraints.
    Expected input is x of shape [batch_size,npendulums,ndims].

    Input:
        l: A torch tensor with the length of the different pendulums, can also be a single number if all pendulums have the same length.
        position_idx: gives the indices for x and y coordinates in ndims.
    """
    def __init__(self,l,tol,niter,position_idx=[0,1],velocity_idx=[2,3],debug_folder=None):
        super(MultiPendulum, self).__init__(tol,niter,debug_folder=debug_folder)
        self.l = l
        self.position_idx = position_idx
        self.velocity_idx = velocity_idx
        return

    def delta_r(self,r):
        """
        Computes a vector from each pendulum to the next, including origo.
        """
        dr_0 = r[:,0]
        dr_i = r[:,1:] - r[:,:-1]
        dr = torch.cat((dr_0[:,None],dr_i),dim=1)
        return dr

    def extract_positions(self,x):
        """
        Extracts positions, r, from x
        """
        r = x[:,:,self.position_idx]
        return r

    def extract_velocity(self,x):
        """
        Extracts velocities, v, from x
        """
        v = x[:,:,self.velocity_idx]
        return v


    def insert_positions(self,r,x_template):
        """
        Inserts, r, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:,:,self.position_idx] = r
        return x

    def debug(self, x,c,extra='',idx=None):
        r = self.extract_positions(x)
        v = self.extract_velocity(x)
        if idx is None:
            idx = torch.argmax(c.mean(dim=1))
        debug_file = f"{self.debug_folder}/{datetime.now():%H_%M_%S.%f}_{extra}.png"
        plot_pendulum_snapshot_custom(r[idx].detach().cpu(), v[idx].detach().cpu(), file=debug_file, fighandler=None, color='red')


    def constraint(self,x):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = |r_i - r_{i-1}| - l_i,   i=1,n
        """
        r = self.extract_positions(x)
        dr = self.delta_r(r)
        drnorm = torch.norm(dr, dim=-1)
        c = drnorm - self.l
        return c

    def jacobian_transpose_times_constraint(self,x,c):
        """
        Computes the Jacobian transpose times the constraints.

        J^Tc =  (c_1 d_1 - c_2 d_2)
                (c_2 d_2 - c_3 d_3)
                        ...
                (c_{n-1} d_{n-1} - c_n d_n)
                (c_n d_n)
        where
        d_i = \frac{r_i - r_{i-1}}{|r_i - r_{i-1}|}

        Note that we do not even have to create this function, since we could also just have let pytorch autograd library do all this.
        In fact you can delete this function and the code will still run since the autograd version is made in the template code, and will in that case just take over the computation.
        """
        r = self.extract_positions(x)
        npend = r.shape[1]
        diffr = self.delta_r(r)
        rnorm = diffr / torch.norm(diffr,dim=-1,keepdim=True)
        dr = torch.zeros_like(r)
        for i in range(npend-1):
            dr[:, i, :] = c[:, i][:, None] * rnorm[:, i, :] - c[:, i+1][:, None] * rnorm[:, i+1, :]
        dr[:,-1,:] = c[:, -1][:, None] * rnorm[:, -1, :]
        dx = self.insert_positions(dr,x)
        return dx



class Water(ConstraintTemplate):
    """
    This constraint function applies constraints to water molecules.
    Expected input is x of shape [batch_size,nwater,ndims].

    Input:
        l: A torch tensor with the binding lengths of the different bonds, can also be a single number if all bonds have the same length.
        position_idx: gives the indices for x,y,z coordinates in ndims for the different particles.
    """
    def __init__(self,l,tol,niter,position_idx=[0,1,2,3,4,5,6,7,8]):
        super(Water, self).__init__(tol,niter)
        self.l = l
        self.position_idx = position_idx
        return

    def extract_positions(self,x):
        """
        Extracts positions, r, from x
        """
        r = x[:,:,self.position_idx]
        r = r.view(r.shape[0],r.shape[1],-1,3)
        return r

    def insert_positions(self,r,x_template):
        """
        Inserts, r, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:,:,self.position_idx] = r.view(r.shape[0],r.shape[1],-1)
        return x

    def delta_r(self,r):
        """
        """
        dr1 = r[:,:,0] - r[:,:,1]
        dr2 = r[:,:,1] - r[:,:,2]
        dr3 = r[:,:,2] - r[:,:,0]
        dr = torch.cat((dr1[:,:,None,:],dr2[:,:,None,:],dr3[:,:,None,:]),dim=2)
        return dr


    def constraint(self,x):
        """
        Computes the constraints.

        For a water molecule the constraint can be given as:
        c_1 = |r_1 - r_2| - l_1
        c_2 = |r_2 - r_3| - l_2
        c_3 = |r_3 - r_1| - l_3
        """
        r = self.extract_positions(x)
        dr = self.delta_r(r)
        drnorm = torch.norm(dr, dim=-1)
        c = drnorm - self.l
        return c

