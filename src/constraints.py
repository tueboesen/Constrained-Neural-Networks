import inspect
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import grad

from src.vizualization import plot_water, plot_pendulum_snapshot_custom
from abc import ABC, abstractmethod

def load_constraints(con,con_type,con_variables=None,rscale=1,vscale=1,device='cpu',debug_folder=None):
    """
    This is a wrapper function for loading constraints.
    """
    if con == 'water':
        r0 = con_variables['r0']
        r1 = con_variables['r1']
        r2 = con_variables['r2']
        l = torch.tensor([r0/rscale,r1/rscale,r2/rscale],device=device)
        niter = 100
        shape_transform = 1 #This is the dimensions of coupled constraints, for the water molecules this should be 1 since they are not coupled.
        con_fnc = Water(l,tol=5e-4,niter=niter,scale=rscale,shape_transform=shape_transform)
    elif con == 'n-pendulum':
        if con_type == 'stabhigh':
            niter = 1
        else:
            niter = 200
        converged_acc = 1e-4
        L = torch.tensor(con_variables['L'][0])
        con_fnc = MultiPendulum(L,niter=niter,tol=converged_acc,debug_folder=debug_folder)
    elif con == 'n-pendulum-with-energy':
        if con_type == 'stabhigh':
            niter = 1
        else:
            niter = 500
        converged_acc = 1e-3
        L = torch.tensor(con_variables['L'][0])
        g = 9.82
        m = 1
        E0 = torch.tensor(0.0) # We have fixed our pendulum system such that the energy of it should always be zero.
        con_fnc = MultiPendulumEnergyConserved(L,niter=niter,tol=converged_acc,m=m,g=g,E0=E0,debug_folder=debug_folder)
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
    elif con == 'n-pendulum' or con == 'n-pendulum-seq' or con == 'n-pendulum-seq-start' or con == 'n-pendulum-with-energy':
        cv['L'] = model_specific['L']
    else:
        NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con, data_type, inspect.currentframe().f_code.co_name))
    return cv

class ConstraintTemplate(ABC):
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

    def __init__(self, tol,niter,sanity_check_upon_first_run=True,debug_folder=None,use_newton_steps=False,mode='gradient_descent',scale=1,shape_transform=None):
        super(ConstraintTemplate, self).__init__()
        self.tol = tol
        self.n = niter
        self.sanity_check = sanity_check_upon_first_run
        self.debug_folder=debug_folder
        self.use_newton_steps = use_newton_steps
        self.mode = mode
        self.scale = scale
        self.shape_transform = shape_transform
        return

    def debug(self,x):
        raise NotImplementedError(
            "Debug function has not been implemented for {:}".format(self._get_name()))

    @abstractmethod
    def constraint(self,x,rescale=False):
        raise NotImplementedError(
            "Constraint function has not been implemented for {:}".format(self._get_name()))

    def jacobian_transpose_times_constraint(self,x,c):
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
        return JTc

    def jacobian_transpose_times_constraint2(self,x,c):
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
        return JTc


    def newton_step_simple(self,x,c,K=None):
        """
        """
        J = torch.autograd.functional.jacobian(self.constraint, x)
        JK = J @ K
        J2 = J.view(-1, *(J.size()[3:]))
        J2 = J2.view(J2.size(0), -1)
        JK2 = JK.view(-1, *(JK.size()[3:]))
        JK2 = JK2.view(JK2.size(0), -1)

        if K is None:
            B = J2 @ J2.T
        else:
            B = JK2 @ JK2.T
        Binv = torch.linalg.inv(B)
        dx = c[:,:,0] @ Binv @ J2
        dx = dx.view(x.shape)
        return dx

    def newton_step(self,x,c,K):
        """
        https://stackoverflow.com/questions/63559139/efficient-way-to-compute-jacobian-x-jacobian-t
        Do something like the above
        """
        J = torch.autograd.functional.jacobian(self.constraint, x)
        B = J @ K @ K.T @ J.T
        Binv = torch.linalg.inv(B)
        dx = J.T @ Binv @ c
        return dx



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


    def conjugate_gradient(self,y,K=None,weight=1,debug=False):
        """
        based on non-linear CG found in
        https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        on page 52
        """
        def f(x):
            c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
            c = c[0]
            y = c.T @ c
            return y


        jmax = 10
        n_restart = 5
        epsilon = 1e-4
        i = 0
        k = 0
        if K is None:
            x = y
        else:
            x = y @ K.T

        c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
        reg = c_error_mean
        reg2 = (c * c).mean()
        xshape = x.shape
        df = torch.autograd.functional.jacobian(f,x).view(-1,1)
        r = - df
        d = r.clone()
        delta = r.T @ r
        delta0 = delta.clone()
        while i < self.n and c_error_max > self.tol:
            j = 0
            deltad = d.T @ d
            while True:
                ddf = torch.autograd.functional.hessian(f,x).view(r.shape[0],r.shape[0])
                alpha = r.T @ d / (d.T @ ddf @ d)
                x = x + alpha * d.view(x.shape)
                c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
                j += 1
                if j >= jmax or alpha**2 * deltad <= epsilon**2 or c_error_max < self.tol:
                    break
                df = torch.autograd.functional.jacobian(f, x).view(-1, 1)
                r = - df
                delta_old = delta
                delta =  r.T @ r
                # print(f"{delta[0, 0]:2.2e},{c_error_max:2.2e}")
                beta = delta / delta_old
                d = r + beta * d
                k += 1
                if k == n_restart or r.T @ d <= 0:
                    d = r
                    k = 0
            i += 1
        # print(f"{delta[0,0]:2.2e},{c_error_max:2.2e}")
        print(f"{i} - {c_error_max:2.2e}")
        return x, reg, reg2

    def gradient_descent(self,y,project,uplift,K=None,weight=1,debug=False,):
        """
        This function has squeezed out the batch dimension, which allows us to use the constraints as written in the paper, for a batch version everything needs to be flipped.
        """
        problems = False
        for j in range(self.n):
            # if K is not None:
            #     x = y @ K.T
            # else:
            #     x = y
            x = project(y)
            c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
            if j == 0:
                reg = c_error_mean
                reg2 = (c * c).mean()
            if c_error_max < self.tol:
                break
            if debug:
                self.debug(x, c, extra=j)
            if self.use_newton_steps:
                dx = self.newton_step_simple(x, c, K)
            else:
                dx = self.jacobian_transpose_times_constraint(x,c)
                # dx = self.jacobian_transpose_times_constraint2(x,c)
            # dx2 = self.newton_step_simple(x,c,K)
            dx = weight * dx
            dy = uplift(dx)
            # if K is not None:
            #     dy = dx @ K
            # else:
            #     dy = dx
            if j == 0:
                alpha = 1.0 / dy.norm(dim=-1).mean()
            lsiter = 0
            while True:
                y_try = y - alpha * dy
                x_try = project(y_try)
                # if K is not None:
                #     x_try = y_try @ K.T
                # else:
                #     x_try = y_try
                c_try, c_error_mean_try, c_error_max_try = self.compute_constraint_violation(x_try)
                if c_error_max_try < c_error_max:
                    break
                alpha = alpha / 2
                lsiter = lsiter + 1
                if alpha == 0:
                    # if c_error_max > 1e-2 and debug is False:
                    #     self.gradient_descent(y_org, project, uplift, weight, debug=True)
                    return y, reg, reg2, True
            if c_error_max_try < c_error_max:
                y = y - alpha * dy
            if lsiter == 0:
                alpha = alpha * 1.5
        if j+1 >= self.n:
            problems = True
        return y, reg, reg2, problems

    def gradient_descent_batch(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug_idx=None):
        # y_org = y.clone()
        nb = y.shape[0]
        alpha = torch.ones(nb,device=y.device)
        j = 0
        while True:
            idx_all = torch.arange(nb,device=y.device)
            x = project(y)
            c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
            if j == 0:
                reg = c_error_mean
                reg2 = (c*c).mean()
            # cm = c.abs().max(dim=2)[0].max(dim=2)[0]
            cm = c.abs().amax(dim=(1,2))
            M = cm > self.tol
            idx = idx_all[M]
            if len(idx) == 0:
                break
            if debug_idx is not None:
                self.debug(x, c, extra=j, idx=debug_idx)
            dx = self.jacobian_transpose_times_constraint(x[idx],c[idx])
            dx = weight[idx] * dx
            dy = uplift(dx)
            lsiter = torch.zeros(len(idx),device=y.device)
            while True:
                y_try = y[idx] - alpha[idx,None,None] * dy
                x_try = project(y_try)
                c_try, c_error_mean_try, c_error_max = self.compute_constraint_violation(x_try)
                # cm_try = c_try.abs().max(dim=1)[0].max(dim=1)[0]
                cm_try = c_try.abs().amax(dim=(1, 2))
                M_try = cm_try <= cm[idx]
                # print(int(lsiter.max().item()), M_try.sum().item(), M_try.shape[0])
                if M_try.all():
                    break
                idx_sel = idx[~M_try]
                alpha[idx_sel] = alpha[idx_sel] / 2
                lsiter[~M_try] = lsiter[~M_try] + 1
                if lsiter.max() > 100:
                    break
            M_increase = lsiter == 0
            idx_sel = idx[M_increase]
            ysel = y[idx] - alpha[idx,None,None] * dy
            alpha[idx_sel] = alpha[idx_sel] * 1.5
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
                # print("Projection failed")
                break
        return y, reg, reg2

    def compute_constraint_violation(self, x):
        """
        Computes the constraint violation.
        """
        c = self.constraint(x,rescale=True)
        cabs = torch.abs(c)
        c_error_mean = torch.mean(cabs)
        c_error_max = torch.max(cabs)
        return c,c_error_mean, c_error_max

    def __call__(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1, use_batch=True,K=None):
        if self.shape_transform is not None:
            y_shape = y.shape
            y_shape_new = torch.Size([y_shape[0]*y_shape[1],self.shape_transform,-1])
            y = y.view(y_shape_new)
            weight = weight.view(y_shape_new)
        if use_batch:
            y, reg, reg2 = self.gradient_descent_batch(y,project,uplift,weight)
        else:
            nb = y.shape[0]
            y_new = []
            reg = 0
            reg2 = 0
            for i in range(nb):
                if self.mode == 'gradient_descent':
                    tmp, regi,regi2,problems = self.gradient_descent(y[i:i+1],project,uplift, K, weight[i:i+1])
                    if problems:
                        tmp, regi, regi2, problems = self.gradient_descent(y[i:i + 1], project, uplift, K, weight[i:i + 1])
                elif self.mode == 'cg':
                    tmp, regi,regi2 = self.conjugate_gradient(y[i:i+1],K=K,weight=weight[i:i+1])
                else:
                    raise NotImplementedError("the mode you have selected is not implemented yet")
                y_new.append(tmp)
                reg = reg + regi
                reg2 = reg2 + regi2
            y = torch.cat(y_new, dim=0)
        if self.shape_transform is not None:
            y = y.view(y_shape)


        return y, reg, reg2

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


    def constraint(self,x,rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = |r_i - r_{i-1}| - l_i,   i=1,n
        """
        r = self.extract_positions(x)
        dr = self.delta_r(r)
        if rescale:
            dr = dr * self.scale
        drnorm = torch.norm(dr, dim=-1)
        c = drnorm - self.l
        return c[:,:,None]

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
            dr[:, i, :] = c[:, i,0][:, None] * rnorm[:, i, :] - c[:, i+1,0][:, None] * rnorm[:, i+1, :]
        dr[:,-1,:] = c[:, -1,0][:, None] * rnorm[:, -1, :]
        dx = self.insert_positions(dr,x)
        return dx




class MultiPendulumEnergyConserved(ConstraintTemplate):
    """
    This constraint function applies multi-body pendulum constraints.
    Expected input is x of shape [batch_size,npendulums,ndims].

    Input:
        l: A torch tensor with the length of the different pendulums, can also be a single number if all pendulums have the same length.
        position_idx: gives the indices for x and y coordinates in ndims.
    """
    def __init__(self,l,tol,niter,E0,m,g,position_idx=[0,1],velocity_idx=[2,3],debug_folder=None):
        super(MultiPendulumEnergyConserved, self).__init__(tol,niter,debug_folder=debug_folder)
        self.l = l
        self.position_idx = position_idx
        self.velocity_idx = velocity_idx
        self.E0 = E0
        self.m = m
        self.g = g
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


    def constraint(self,x,rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the length constraint can be given as:
        c_i = |r_i - r_{i-1}| - l_i,   i=1,n
        and the energy conservation constraint is given as:
        c_i = m_i * (0.5 * v_i**2 + g*y) - E0
        """
        r = self.extract_positions(x)
        v = self.extract_velocity(x)
        dr = self.delta_r(r)
        if rescale:
            dr = dr * self.scale
        drnorm = torch.norm(dr, dim=-1)
        c_len = drnorm - self.l

        c_energy = torch.sum(self.m * (0.5* torch.sum(v*v,dim=-1) + self.g * r[:,:,-1]) - self.E0,dim=-1)
        c = torch.cat([c_len,c_energy[:,None]],dim=1)
        return c[...,None]




class Water(ConstraintTemplate):
    """
    This constraint function applies constraints to water molecules.
    Expected input is x of shape [batch_size,nwater,ndims].

    Input:
        l: A torch tensor with the binding lengths of the different bonds, can also be a single number if all bonds have the same length.
        position_idx: gives the indices for x,y,z coordinates in ndims for the different particles.
    """
    def __init__(self,l,tol,niter,position_idx=[0,1,2,3,4,5,6,7,8],scale=1,shape_transform=None):
        super(Water, self).__init__(tol,niter,scale=scale,shape_transform=shape_transform)
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


    def constraint(self,x,extract_position=True, rescale=False):
        """
        Computes the constraints.

        For a water molecule the constraint can be given as:
        c_1 = |r_1 - r_2| - l_1
        c_2 = |r_2 - r_3| - l_2
        c_3 = |r_3 - r_1| - l_3
        """
        if extract_position:
            r = self.extract_positions(x)
        else:
            r = x
        dr = self.delta_r(r)
        if rescale:
            l = self.l * self.scale
            dr = dr * self.scale
        else:
            l = self.l
        drnorm = torch.norm(dr, dim=-1)
        # drnorm2 = torch.norm(dr, dim=-1)

        c = drnorm - l
        return c


if __name__ == '__main__':

    """We test that the water molecule constraint is sensible."""

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=10)
    l0 = torch.tensor(0.957)
    l1 = torch.tensor(1.513)
    l2 = torch.tensor(0.957)

    alpha_rad = torch.acos((l0**2+l2**2-l1**2)/(2*l0*l2))
    alpha_degree = alpha_rad / torch.pi * 180

    l = torch.tensor([l0 , l1 , l2 ])
    beta = (180-alpha_degree)/2  # degrees
    b = torch.tensor(beta / 180 * torch.pi)

    r1 = torch.tensor([[0.0, 0.0, 0.0], [torch.cos(b) * l0, torch.sin(b) * l0, 0.0], [-torch.cos(b) * l0, torch.sin(b) * l0, 0.0]])
    r = r1[None, None, :]

    con_fnc = Water(l, tol=1e-2, niter=100)
    c = con_fnc.constraint(r,extract_position=False)
    assert c.allclose(torch.tensor(0.0))

#
# class ConstraintTemplate(nn.Module):
#     """
#     This is the template class for all constraints.
#     Each constraint should have this class as their parent.
#
#     When making a constraint class, you only need to define an appropriate constraint function.
#
#     If you supply high dimensional data y, the data needs to be supplied with a torch tensor K, such that x = y @ K and y = x @ K.T
#     If you do not supply K, then K will essentially be an identity matrix, such that x=y
#
#     Input:
#         tol: the error tolerance for early stopping of the gradient descent operation.
#         niter: Maximum number of gradient descent steps taken.
#     """
#
#     def __init__(self, tol,niter,sanity_check_upon_first_run=True,debug_folder=None):
#         super(ConstraintTemplate, self).__init__()
#         self.tol = tol
#         self.n = niter
#         self.sanity_check = sanity_check_upon_first_run
#         self.debug_folder=debug_folder
#         return
#
#     def debug(self,x):
#         raise NotImplementedError(
#             "Debug function has not been implemented for {:}".format(self._get_name()))
#
#     def constraint(self,x):
#         raise NotImplementedError(
#             "Constraint function has not been implemented for {:}".format(self._get_name()))
#
#     def jacobian_transpose_times_constraint(self,x,c):
#         _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
#         return JTc
#
#     def newton_step_simple(self,x,c,K):
#         J = torch.autograd.functional.jacobian(self.constraint, x)
#         B = J @ K @ K.T @ J.T
#         B2 = J.T @ K.T @ K.T @ J
#         Binv = torch.linalg.inv(B)
#         dx = J.T @ Binv @ c
#         return dx
#
#     def newton_step(self,x,c,K):
#         """
#         https://stackoverflow.com/questions/63559139/efficient-way-to-compute-jacobian-x-jacobian-t
#         Do something like the above
#         """
#         J = torch.autograd.functional.jacobian(self.constraint, x)
#         B = J @ K @ K.T @ J.T
#         Binv = torch.linalg.inv(B)
#         dx = J.T @ Binv @ c
#         return dx
#
#
#
#     def constrain_stabilization(self, y,project=nn.Identity(),uplift=nn.Identity(),weight=1):
#         """
#         Calculates constraint stabilization as dy = J^T c
#         """
#         x = project(y)
#         c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
#         dx = self.jacobian_transpose_times_constraint(x,c)
#         dx = weight * dx
#         dy = uplift(dx)
#         return dy
#
#
#     # def gradient_descent(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug=False):
#     def gradient_descent(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug=False,K=None):
#         # y_org = y.clone()
#         for j in range(self.n):
#             # x = y @ self.K
#             x = project(y)
#             c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
#             if j == 0:
#                 reg = c_error_mean
#                 reg2 = (c * c).mean()
#             if c_error_max < self.tol:
#                 break
#             if debug:
#                 self.debug(x, c, extra=j)
#             dx = self.jacobian_transpose_times_constraint(x,c)
#             dx2 = self.newton_step_simple(x,c,K)
#             dx = weight * dx
#             dy = uplift(dx)
#             if j == 0:
#                 alpha = 1.0 / dy.norm(dim=-1).mean()
#             lsiter = 0
#             while True:
#                 y_try = y - alpha * dy
#                 x_try = project(y_try)
#                 c_try, c_error_mean_try, c_error_max_try = self.compute_constraint_violation(x_try)
#                 if c_error_max_try < c_error_max:
#                     break
#                 alpha = alpha / 2
#                 lsiter = lsiter + 1
#                 if alpha == 0:
#                     # if c_error_max > 1e-2 and debug is False:
#                     #     self.gradient_descent(y_org, project, uplift, weight, debug=True)
#                     return y
#             if lsiter == 0 and c_error_max_try > self.tol:
#                 alpha = alpha * 1.5
#             if c_error_max_try < c_error_max:
#                 y = y - alpha * dy
#         if j+1 >= self.n:
#             print("problems detected!")
#             # self.gradient_descent(y_org, project, uplift, weight, extra='n_exceeded', debug=True)
#         return y, reg,reg2
#
#
#     def gradient_descent_batch(self,y,project=nn.Identity(),uplift=nn.Identity(),weight=1,debug_idx=None):
#         # y_org = y.clone()
#         nb = y.shape[0]
#         alpha = torch.ones(nb,device=y.device)
#         j = 0
#         while True:
#         # for j in range(self.n):
#             idx_all = torch.arange(nb)
#             x = project(y)
#             c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
#             if j == 0:
#                 reg = c_error_mean
#                 reg2 = (c*c).mean()
#             cm = c.abs().sum(dim=2).max(dim=1)[0]
#             M = cm > self.tol
#             idx = idx_all[M]
#             if len(idx) == 0:
#                 break
#             if debug_idx is not None:
#                 self.debug(x, c, extra=j, idx=debug_idx)
#             dx = self.jacobian_transpose_times_constraint(x[idx],c[idx])
#             dx = weight[idx] * dx
#             dy = uplift(dx)
#             lsiter = torch.zeros(len(idx))
#             while True:
#                 y_try = y[idx] - alpha[idx,None,None] * dy
#                 x_try = project(y_try)
#                 c_try, c_error_mean_try, c_error_max = self.compute_constraint_violation(x_try)
#                 cm_try = c_try.abs().sum(dim=2).max(dim=1)[0]
#                 M_try = cm_try <= cm[idx]
#                 if M_try.all():
#                     break
#                 idx_sel = idx[~M_try]
#                 alpha[idx_sel] = alpha[idx_sel] / 2
#                 lsiter[~M_try] = lsiter[~M_try] + 1
#                 if lsiter.max() > 100:
#                     break
#             M_increase = lsiter == 0
#             idx_sel = idx[M_increase]
#             alpha[idx_sel] = alpha[idx_sel] * 1.5
#             ysel = y[idx] - alpha[idx,None,None] * dy
#             yall = []
#             count = 0
#             for i in range(nb):
#                 if M[i]:
#                     yall.append(ysel[count])
#                     count = count + 1
#                 else:
#                     yall.append(y[i])
#             y = torch.stack(yall,dim=0)
#             j += 1
#             if j > self.n:
#                 # print(f"{cm.max()}")
#                 # if debug_idx is None:
#                 #     self.gradient_descent_batch(y_org,project,uplift,weight,debug_idx=cm.argmax())
#                 # x = project(y)
#                 # x_org = project(y_org)
#                 # c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
#                 # c_org, c_org_error_mean, c_org_error_max = self.compute_constraint_violation(x_org)
#                 # dx = self.jacobian_transpose_times_constraint(x, c)
#                 # dx2 = self.jacobian_transpose_times_constraint_backup(x, c)
#                 # self.debug(x,c)
#                 # self.debug(x_org,c_org,extra='org')
#                 # assert torch.allclose(dx,dx2,rtol=1e-7), "The jacobian is wrong!"
#
#                 break
#         return y, reg, reg2
#
#     def compute_constraint_violation(self, x):
#         """
#         Computes the constraint violation.
#         """
#         c = self.constraint(x)
#         cabs = torch.abs(c)
#         c_error_mean = torch.mean(cabs)
#         c_error_max = torch.max(torch.abs(cabs))
#         return c,c_error_mean, c_error_max
#
#     def forward(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1, use_batch=False,K=None):
#         if use_batch:
#             y, reg, reg2 = self.gradient_descent_batch(y,project,uplift,weight)
#         else:
#             nb = y.shape[0]
#             y_new = []
#             reg = 0
#             reg2 = 0
#             for i in range(nb):
#                 tmp, regi,regi2 = self.gradient_descent(y[i:i+1], project, uplift, weight[i:i+1],K=K)
#                 y_new.append(tmp)
#                 reg = reg + regi
#                 reg2 = reg2 + regi2
#             y = torch.cat(y_new, dim=0)
#         return y, reg, reg2
