import numbers

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.constraints import ConstraintTemplate


def generate_constraints_minimizer(c,con_fnc):
    """
    This is a wrapper function for generating/loading constraints.
    """
    c_dict = OmegaConf.to_container(c)
    name = c_dict.pop("name", None)
    name = name.lower()


    if name == 'gradient_descent':
        min_con_fnc = GradientDescentFast(constraint=con_fnc,**c_dict)
    else:
        raise NotImplementedError(f"The constraint {name} has not been implemented.")
    return min_con_fnc


class ProjectUpliftFromMatrix:
    """
    If a matrix is given for projection and uplifting, then we convert the matrix multiplication operation (from the right) into a function.
    """
    def __init__(self,K: torch.Tensor):
        self.K = K

    def project(self,y):
        return y @ self.K.T

    def uplift(self,x):
        return x @ self.K


class MinimizationTemplate(ABC, nn.Module):
    """
    This is a template for minimizing constraint violations.
    Essentially you can think of this as a standard nonlinear optimization methods, with the exception that they include a projection and uplifting function or matrix.

    If you supply high dimensional data y, the data needs to be supplied with a torch tensor K, such that:
        the projection is given as: x = y @ K
        the uplifting is given as:  y = x @ K.T
    Alternatively you can supply a projection / uplifting function.
    If both as supplied, the matrix will be used and converted into appropriate projection and uplifting functions.
    If neither is supplied, the projection and uplifting functions will be identity functions.

    Input:
        constraint: Is the constraint function we wish to minimize our data across.
        max_iter: Is the maximum number of minimization iteration we apply
        rel_tol: Is the relative tolerance needed before stopping early.
        abs_tol: Is the absolute tolerance needed before stopping early.
        max_iter_linesearch: Is the max number of linesearch iteration applied during each iteration.
    """
    def __init__(self,constraint: ConstraintTemplate, max_iter: int,rel_tol: float,abs_tol: float,max_iter_linesearch=20):
        super(MinimizationTemplate, self).__init__()
        self.c = constraint
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.max_iter_linesearch = max_iter_linesearch
        return
    def __call__(self, y, K=None, project_fnc=None, uplift_fnc=None, weight=1):
        project_fnc, uplift_fnc = self.determine_proj_uplift_method(K,project_fnc,uplift_fnc)
        if isinstance(weight, numbers.Number):
            # weight = weight * torch.ones(y.shape[:-1], device=y.device)
            weight = weight * torch.ones_like(y)
        y = self._minimize(y,project_fnc,uplift_fnc,weight)
        return y

    def determine_proj_uplift_method(self,K,project_fnc,uplift_fnc):
        if K is not None:
            PU = ProjectUpliftFromMatrix(K)
            project_fnc = PU.project
            uplift_fnc = PU.uplift
        else:
            if (project_fnc is None) or (uplift_fnc is None):
                assert (project_fnc is None) and (uplift_fnc is None)
                project_fnc = nn.Identity()
                uplift_fnc = nn.Identity()
        return project_fnc, uplift_fnc

    @abstractmethod
    def _minimize(self,y,project_fnc,uplift_fnc,weight):
        raise NotImplementedError(
            "minimization function has not been implemented for {:}".format(self._get_name()))

class GradientDescentFast(MinimizationTemplate):
    """
    A batched gradient descent algorithm which enables minimization of high dimensional data constrained in low dimensions.
    If project_fnc and uplift_fnc are identity functions, this gradient descent algorithm will work as a standard gradient descent algorithm.
    """
    def _minimize(self,y,project_fnc,uplift_fnc,weight):
        nb = y.shape[0]
        alpha = torch.ones(nb, device=y.device)
        j = 0
        while True:
            idx_all = torch.arange(nb, device=y.device)
            x = project_fnc(y)
            c, c_error_mean, c_error_max = self.c.constraint_violation(x)
            cm_norm = c.abs().amax(dim=(1, 2))
            M = cm_norm > self.rel_tol
            idx = idx_all[M]
            if len(idx) == 0:
                break
            else:
                c = self.c._constraint(x)
                cm = c.abs().mean(dim=(1, 2))
            dx = self.c._jacobian_transpose_times_constraint(x[idx], c[idx])
            # dx = weight[idx][:,:,None] * dx
            dx = weight[idx].view(dx.shape) * dx
            dy = uplift_fnc(dx)
            lsiter = torch.zeros(len(idx), device=y.device)
            while True:
                y_try = y[idx] - alpha[idx, None, None] * dy
                x_try = project_fnc(y_try)
                c_try = self.c._constraint(x_try)
                cm_try = c_try.abs().mean(dim=(1, 2))
                M_try = cm_try <= cm[idx]
                if M_try.all():
                    break
                idx_sel = idx[~M_try]
                alpha[idx_sel] = alpha[idx_sel] / 2
                lsiter[~M_try] = lsiter[~M_try] + 1
                if lsiter.max() > self.max_iter_linesearch:
                    break
            M_increase = lsiter == 0
            idx_sel = idx[M_increase]
            ysel = y[idx] - alpha[idx, None, None] * dy
            alpha[idx_sel] = alpha[idx_sel] * 1.5
            yall = []
            count = 0
            for i in range(nb):
                if M[i]:
                    yall.append(ysel[count])
                    count = count + 1
                else:
                    yall.append(y[i])
            y = torch.stack(yall, dim=0)
            j += 1
            if alpha.min() < self.rel_tol:
                break
            if j > self.max_iter:
                break
        return y


class ConjugateGradient(MinimizationTemplate):
    """
    based on non-linear CG found in
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    on page 52
    Note that this version is lacking and running 1 sample at a time making it extremely slow.
    """
    def _minimize(self,y,project_fnc,uplift_fnc,weight):
        def f(x):
            c = self._constraint(x)
            return c.abs().mean()

        raise Warning("This function is not fully implemented and should not be used.")

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

        # x_org = x.clone()

        df = torch.autograd.functional.jacobian(f, x, strict=True, create_graph=True).view(-1, 1)
        r = - df
        d = r.clone()
        delta_new = r.T @ r
        delta0 = delta_new.clone()
        while i < self.max_iter and delta_new > epsilon ** 2 * delta0:
            j = 0
            deltad = d.T @ d
            c = f(x)
            alpha = torch.min(torch.tensor(1), c.abs().mean())
            while True:
                # ddf = torch.autograd.functional.hessian(f, x, create_graph=True, strict=False).view(r.shape[0],r.shape[0])
                # alpha = - (df.T @ d) / (d.T @ ddf @ d)
                x_try = x + alpha * d.view(x.shape)
                c_try = f(x_try)
                if c.abs().mean() > c_try.abs().mean() or j > jmax:
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
        # print(f"{i},  {c.abs().mean():2.2e}")
        if c.abs().mean() > tol:
            print("here")
        return x
