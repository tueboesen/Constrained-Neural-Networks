import inspect
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import grad

from src.vizualization import plot_water, plot_pendulum_snapshot_custom
from abc import ABC, abstractmethod

def generate_constraints(c,rscale=1,vscale=1):
    """
    This is a wrapper function for generating/loading constraints.
    #TODO add a minimum change amount to constraints which might enable non-convex constraints to still work
    """
    if c.name == 'multibodypendulum':
        con_fnc = MultiBodyPendulum(max_iter=c.max_iter, tol=c.tolerance, include_second_order=c.include_second_order_constraints)
    else:
        raise NotImplementedError(f"The constraint {c.name} has not been implemented.")
    return con_fnc


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

    def __init__(self, tol,niter,sanity_check_upon_first_run=True,debug_folder=None,use_newton_steps=False,mode='cg',scale=1,shape_transform=None):
    # def __init__(self, tol,niter,sanity_check_upon_first_run=True,debug_folder=None,use_newton_steps=False,mode='gradient_descent',scale=1,shape_transform=None):
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
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c,strict=True,create_graph=True)
        return JTc

    def jacobian_transpose_times_constraint2(self,x,c):
        _, JTc = torch.autograd.functional.vjp(self.constraint,x,c)
        return JTc

    def constraint_violation(self,x,rescale=False):
        raise NotImplementedError(
            "constraint_violation function has not been implemented for {:}".format(self._get_name()))
    def second_order_constraint(self,func,x,v):
        """
        The second order constraint is the constraint

        \dot{c} = C(x) \cdot \dot{x} = 0

        where C(x) = \frac{\partial c}{\partial x}

        """
        _, Jv = torch.autograd.functional.jvp(func, x, v,create_graph=True,strict=True)
        return Jv



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
        c = self.constraint(x)
        # c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
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
            c = self.constraint(x)
            return c.abs().mean()

        imax = 1000
        tol = 1e-5
        jmax = 100
        n_restart = 10
        epsilon = 1e-4
        i = 0
        k = 0

        if K is None:
            x = y
        else:
            x = y @ K.T

        x_org = x.clone()

        df = torch.autograd.functional.jacobian(f, x, strict=True, create_graph=True).view(-1, 1)
        r = - df
        d = r.clone()
        delta_new = r.T @ r
        delta0 = delta_new.clone()
        while i < imax and delta_new > epsilon ** 2 * delta0:
            j = 0
            deltad = d.T @ d
            c = f(x)
            alpha = torch.min(torch.tensor(1),c.abs().mean())
            while True:
                # ddf = torch.autograd.functional.hessian(f, x, create_graph=True, strict=False).view(r.shape[0],r.shape[0])
                # alpha = - (df.T @ d) / (d.T @ ddf @ d)
                x_try = x + alpha * d.view(x.shape)
                c_try = f(x_try)
                if c.abs().mean() > c_try.abs().mean() or j>jmax:
                    # print(f"{j}  {alpha.item():2.2e}  {c.abs().mean():2.2e} -> {c_try.abs().mean():2.2e}")
                    x = x_try
                    break
                else:
                    alpha *= 0.5
                    j += 1
            if c_try.abs().mean() < tol:
                break
            df = torch.autograd.functional.jacobian(f, x, strict=True, create_graph=True).view(-1, 1)
            r = - df
            delta_old = delta_new
            delta_new = r.T @ r
            beta = delta_new / delta_old
            beta = max(beta, 0)
            d = r + beta * d
            k += 1
            if k == n_restart or r.T @ d <= 0:
                d = r
                k = 0
            i += 1
        c = f(x)
        print(f"{i},  {c.abs().mean():2.2e}")
        if c.abs().mean() > tol:
            print("here")
        return x, 0, 0
    #
    # def conjugate_gradient(self,y,K=None,weight=1,debug=False):
    #     """
    #     based on non-linear CG found in
    #     https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    #     on page 52
    #     """
    #     def f(x):
    #         c = self.constraint(x)
    #         return c.abs().mean()
    #
    #     def f2(x):
    #         c = self.constraint(x)
    #         return c
    #
    #     jmax = 10
    #     n_restart = 5
    #     epsilon = 1e-4
    #     i = 0
    #     k = 0
    #     if K is None:
    #         x = y
    #     else:
    #         x = y @ K.T
    #
    #     c = f(x)
    #     df = torch.autograd.functional.jacobian(f,x,strict=True,create_graph=True).view(-1,1)
    #     r = - df
    #     d = r.clone()
    #     delta_new = r.T @ r
    #     # delta0 = delta_new.clone()
    #     while i < self.n and delta_new > self.tol:
    #         j = 0
    #         deltad = d.T @ d
    #         while True:
    #             ddf = torch.autograd.functional.hessian(f,x,create_graph=True,strict=True).view(r.shape[0],r.shape[0])
    #             alpha = - (df.T @ d) / (d.T @ ddf @ d)
    #             x = x + alpha * d.view(x.shape)
    #             c_try = f(x)
    #             print(f"{c.abs().mean():2.2e} -> {c_try.abs().mean()}")
    #             j += 1
    #             if j >= jmax or alpha**2 * deltad <= epsilon**2 or c_try.abs().mean() < self.tol:
    #                 break
    #             df = torch.autograd.functional.jacobian(f, x,strict=True,create_graph=True).view(-1, 1)
    #             r = - df
    #             delta_old = delta_new
    #             delta_new =  r.T @ r
    #             # print(f"{delta[0, 0]:2.2e},{c_error_max:2.2e}")
    #             beta = delta_new / delta_old
    #             d = r + beta * d
    #             k += 1
    #             if k == n_restart or r.T @ d <= 0:
    #                 d = r
    #                 k = 0
    #         i += 1
    #     # print(f"{delta[0,0]:2.2e},{c_error_max:2.2e}")
    #     print(f"{i} - {c.abs().mean():2.2e}")
    #     return x, 0, 0

    def conjugate_gradient_old(self,y,K=None,weight=1,debug=False):
        """
        based on non-linear CG found in
        https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        on page 52
        """
        def f(x):
            c = self.constraint(x)
            return c.abs().mean()

        def f2(x):
            c = self.constraint(x)
            return c

        jmax = 10
        n_restart = 5
        epsilon = 1e-4
        i = 0
        k = 0
        if K is None:
            x = y
        else:
            x = y @ K.T

        # result = minimize(f, x)
        # result = minimize(f, x, method='newton-cg',disp=2,tol=1e-5)
        # result = minimize(f, x, method='newton-cg',tol=1e-5)
        # result = minimize(f, x, method='bfgs',tol=1e-5)
        # result = minimize(f2, x, method='bfgs',tol=1e-5)
        # if result.success:
        #     xnew = result.x
        # # f = self.constraint
        # assert result.success
        c = f(x)
        # cnew = f(xnew)
        # print(f"success: {result.success} nfev:{result.nfev}, nit:{result.nit}, {c:2.2e} -> {cnew:2.2e}")
        # print(f"success: {result.success} ncg:{result.ncg}, nfev:{result.nfev}, nit:{result.nit}, {c:2.2e} -> {cnew:2.2e}")
        df = torch.autograd.functional.jacobian(f,x,strict=True,create_graph=True).view(-1,1)
        r = - df
        d = r.clone()
        delta_new = r.T @ r
        # delta0 = delta_new.clone()
        while i < self.n and delta_new > self.tol:
            j = 0
            deltad = d.T @ d
            while True:
                ddf = torch.autograd.functional.hessian(f,x,create_graph=True,strict=True).view(r.shape[0],r.shape[0])
                alpha = - (df.T @ d) / (d.T @ ddf @ d)
                x = x + alpha * d.view(x.shape)
                c_try = f(x)
                print(f"{c.abs().mean():2.2e} -> {c_try.abs().mean()}")
                j += 1
                if j >= jmax or alpha**2 * deltad <= epsilon**2 or c_try.abs().mean() < self.tol:
                    break
                df = torch.autograd.functional.jacobian(f, x,strict=True,create_graph=True).view(-1, 1)
                r = - df
                delta_old = delta_new
                delta_new =  r.T @ r
                # print(f"{delta[0, 0]:2.2e},{c_error_max:2.2e}")
                beta = delta_new / delta_old
                d = r + beta * d
                k += 1
                if k == n_restart or r.T @ d <= 0:
                    d = r
                    k = 0
            i += 1
        # print(f"{delta[0,0]:2.2e},{c_error_max:2.2e}")
        print(f"{i} - {c.abs().mean():2.2e}")
        return x, 0, 0

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
                cm_norm_init = c.abs().max()
            # cm = c.abs().max(dim=2)[0].max(dim=2)[0]
            cm_norm = c.abs().amax(dim=(1,2))
            M = cm_norm > self.tol
            idx = idx_all[M]
            if len(idx) == 0:
                break
            else:
                c = self.constraint(x)
                cm = c.abs().mean(dim=(1,2))
            if debug_idx is not None:
                self.debug(x, c, extra=j, idx=debug_idx)
            dx = self.jacobian_transpose_times_constraint(x[idx],c[idx])
            dx = weight[idx] * dx
            dy = uplift(dx)
            lsiter = torch.zeros(len(idx),device=y.device)
            while True:
                y_try = y[idx] - alpha[idx,None,None] * dy
                x_try = project(y_try)
                c_try = self.constraint(x_try)
                # cm_try = c_try.abs().max(dim=1)[0].max(dim=1)[0]
                cm_try = c_try.abs().mean(dim=(1,2))
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
        print(f"projection {j} steps. Max violation: {cm_norm_init.item():2.2e} -> {cm_norm.max().item():2.2e}")
        return y, reg, reg2

    def compute_constraint_violation(self, x):
        """
        Computes the constraint violation.
        """
        try:
            c = self.constraint_violation(x,rescale=True)
        except:
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
        if use_batch and self.mode == 'gradient_descent':
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

class MultiBodyPendulum(ConstraintTemplate):
    """
    This constraint function applies multi-body pendulum constraints.
    Expected input is x of shape [batch_size,npendulums,ndims].

    Input:
        l: A torch tensor with the length of the different pendulums, can also be a single number if all pendulums have the same length.
        position_idx: gives the indices for x and y coordinates in ndims.
    """
    def __init__(self,max_iter,tol,include_second_order=False,position_idx=(0,1),velocity_idx=(2,3),debug_folder=None):
        super(MultiBodyPendulum, self).__init__(tol, max_iter, debug_folder=debug_folder)
        self.l = 1
        self.position_idx = position_idx
        self.velocity_idx = velocity_idx
        self.include_second_order = include_second_order
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

    def insert_velocities(self,v,x_template):
        """
        Inserts, v, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:,:,self.velocity_idx] = v
        return x


    def debug(self, x,c,extra='',idx=None):
        r = self.extract_positions(x)
        v = self.extract_velocity(x)
        if idx is None:
            idx = torch.argmax(c.mean(dim=1))
        debug_file = f"{self.debug_folder}/{datetime.now():%H_%M_%S.%f}_{extra}.png"
        plot_pendulum_snapshot_custom(r[idx].detach().cpu(), v[idx].detach().cpu(), file=debug_file, fighandler=None, color='red')

    def constraint_base(self,x,rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = (r_i - r_{i-1})**2 - l_i**2,   i=1,n

        Note that this should not be used when computing the constraint violation since that should be done with the unsquared norm.
        """
        r = self.extract_positions(x)
        dr = self.delta_r(r)
        if rescale:
            dr = dr * self.scale
        # drnorm = torch.norm(dr, dim=-1)
        dr2 = (dr*dr).sum(dim=-1)
        c = dr2 - self.l*self.l
        return c[:,:,None]

    def constraint(self,x,rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = |r_i - r_{i-1}| - l_i,   i=1,n

        Note that this is the unsquared norm, which is not differentiable at r_i = r_{i-1}
        """
        c1 = self.constraint_base(x,rescale=rescale)
        if self.include_second_order:
            v = self.extract_velocity(x)
            r = self.extract_positions(x)
            c2 = self.second_order_constraint(self.constraint_base,r,v)
            # c3 = self.second_order_constraint2(r,v)
            # assert torch.allclose(c2,c3)
            c = torch.cat((c1,c2),dim=-1)
        else:
            c = c1
        return c

    def constraint_violation_base(self,x,rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = ||r_i - r_{i-1}||_2 - l_i,   i=1,n
        """
        r = self.extract_positions(x)
        dr = self.delta_r(r)
        if rescale:
            dr = dr * self.scale
        drnorm = torch.norm(dr, dim=-1)
        c = drnorm - self.l
        return c[:,:,None]

    def constraint_violation(self,x,rescale=False):
        c1 = self.constraint_violation_base(x,rescale=rescale)
        if self.include_second_order:
            v = self.extract_velocity(x)
            r = self.extract_positions(x)
            c2 = self.second_order_constraint(self.constraint_violation_base,r,v)
            # c3 = self.second_order_constraint2(r,v)
            # assert torch.allclose(c2,c3)
            c = torch.cat((c1,c2),dim=-1)
        else:
            c = c1
        return c


    def second_order_constraint(self,func,r,v):
        dr = self.delta_r(r)
        dv = self.delta_r(v)
        c = 2*(dr*dv).sum(dim=-1)

        return c[:,:,None]
    #
    # def jacobian_transpose_times_constraint(self,x,c):
    #     """
    #     Computes the Jacobian transpose times the constraints.
    #
    #     J^Tc =  (c_1 d_1 - c_2 d_2)
    #             (c_2 d_2 - c_3 d_3)
    #                     ...
    #             (c_{n-1} d_{n-1} - c_n d_n)
    #             (c_n d_n)
    #     where
    #     d_i = \frac{r_i - r_{i-1}}{|r_i - r_{i-1}|}
    #
    #     Note that we do not even have to create this function, since we could also just have let pytorch autograd library do all this.
    #     In fact you can delete this function and the code will still run since the autograd version is made in the template code, and will in that case just take over the computation.
    #     """
    #     r = self.extract_positions(x)
    #     npend = r.shape[1]
    #     diffr = self.delta_r(r)
    #     rnorm = diffr / torch.norm(diffr,dim=-1,keepdim=True)
    #     dr = torch.zeros_like(r)
    #     for i in range(npend-1):
    #         dr[:, i, :] = c[:, i,0][:, None] * rnorm[:, i, :] - c[:, i+1,0][:, None] * rnorm[:, i+1, :]
    #     dr[:,-1,:] = c[:, -1,0][:, None] * rnorm[:, -1, :]
    #     dx = self.insert_positions(dr,x)
    #     return dx

