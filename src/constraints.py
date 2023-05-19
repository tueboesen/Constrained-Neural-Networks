import numbers
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def generate_constraints(c):
    """
    This is a wrapper function for generating/loading constraints.
    """

    c_dict = OmegaConf.to_container(c)
    name = c_dict.pop("name", None)

    if name == 'multibodypendulum':
        con_fnc = MultiBodyPendulum(**c_dict)
    elif name == 'water':
        con_fnc = Water(**c_dict)
    elif name == None:
        con_fnc = None
    else:
        raise NotImplementedError(f"The constraint {name} has not been implemented.")
    return con_fnc


class ConstraintTemplate(ABC, nn.Module):
    """
    This is the template class for all constraints.
    Each constraint should have this class as their parent.

    When making a constraint class, you only need to define an appropriate constraint function.

    If you supply high dimensional data y, the data needs to be supplied with a torch tensor K, such that x = y @ K and y = x @ K.T
    If you do not supply K, then K will essentially be an identity matrix, such that x=y

    Input:
        tol: the error tolerance for early stopping of the gradient descent operation.
        niter: Maximum number of gradient descent steps taken.

    The expected input shape of a parameter being minimized is:
    [nb,nc,nv,dim]
    where
    nb is the number of batches (so separate samples that have no constraint connection) Note that this does not have to be the same as regular batch_size
    nc is the number of coupled constraints in a batch
    nv is the number of variables in play (is it position, velocity or is it )
    dim is the number of dimensions for each variable


    We assume particle constraints here. The constraints assume the following:
    [nb,nc,np,dim]
    nb is the number of batches (so separate samples that have no constraint connection) Note that this does not have to be the same as regular batch_size
    nc is the number of coupled constraints in a batch
    np is the number of particles in play (is it position, velocity or is it )
    """

    def __init__(self, tolerance, max_iter, minimizer='gradient_descent', n_constraints=None, scale=1, include_second_order_constraints=False, abs_tol=None):
        super(ConstraintTemplate, self).__init__()
        self.include_second_order_constraints = include_second_order_constraints
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.scale = scale
        self.minimizer = minimizer
        assert minimizer in ['cg', 'gradient_descent']
        self.n_constraints = n_constraints
        if abs_tol is None:
            abs_tol = tolerance * 0.01
        self.abs_tol = abs_tol
        return

    def debug(self, x):
        raise NotImplementedError(
            "Debug function has not been implemented for {:}".format(self._get_name()))

    @abstractmethod
    def _constraint(self, x, rescale=False):
        raise NotImplementedError(
            "Constraint function has not been implemented for {:}".format(self._get_name()))

    def _jacobian_transpose_times_constraint(self, x, c):
        _, JTc = torch.autograd.functional.vjp(self._constraint, x, c, strict=True, create_graph=True)
        return JTc

    def _constraint_violation(self, x, rescale=False):
        raise NotImplementedError(
            "constraint_violation function has not been implemented for {:}".format(self._get_name()))

    def _second_order_constraint(self, func, x, v):
        """
        The second order constraint is the constraint

        \dot{c} = C(x) \cdot \dot{x} = 0

        where C(x) = \frac{\partial c}{\partial x}

        """
        _, Jv = torch.autograd.functional.jvp(func, x, v, create_graph=True, strict=True)
        return Jv

    def constraint_penalty(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1):
        """
        Calculates constraint stabilization as dy = J^T c
        Note that we have added a clamp, which prevents this penalty creating changes larger than 10% of the value of x.
        #TODO the 10% should be added to conf files such that it can be tested for different values, systematically it might be set too low at the moment.
        """
        x = project(y)
        c = self._constraint(x)
        # c, c_error_mean, c_error_max = self.compute_constraint_violation(x)
        dx = self._jacobian_transpose_times_constraint(x, c)
        dx = weight.view(dx.shape) * dx
        clampval = x.abs() * 0.1
        dx = torch.clamp(dx, min=-clampval, max=clampval)
        dy = uplift(dx)
        return dy

    def conjugate_gradient(self, y, K=None, weight=1, debug=False):
        """
        based on non-linear CG found in
        https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        on page 52
        Note that this version is lacking and running 1 sample at a time making it extremely slow.
        """

        def f(x):
            c = self._constraint(x)
            return c.abs().mean()

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
        return x, 0, 0

    def gradient_descent(self, y, project, uplift, K=None, weight=1, debug=False, ):
        """
        This function has squeezed out the batch dimension, which allows us to use the constraints as written in the paper, for a batch version everything needs to be flipped.
        """
        problems = False
        for j in range(self.max_iter):
            # if K is not None:
            #     x = y @ K.T
            # else:
            #     x = y
            x = project(y)
            c, c_error_mean, c_error_max = self.constraint_violation(x)
            if j == 0:
                reg = c_error_mean
                reg2 = (c * c).mean()
            if c_error_max < self.tolerance:
                break
            if debug:
                self.debug(x, c, extra=j)
            if self.use_newton_steps:
                dx = self.newton_step_simple(x, c, K)
            else:
                dx = self._jacobian_transpose_times_constraint(x, c)
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
                c_try, c_error_mean_try, c_error_max_try = self.constraint_violation(x_try)
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
        if j + 1 >= self.max_iter:
            problems = True
        return y, reg, reg2, problems

    def gradient_descent_batch(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1, debug_idx=None):
        """
        A much faster version of gradient_descent which runs on a batch of samples. Note that due to some implementation details the two versions might not produce identical results, but they should be close.
        """
        if isinstance(weight, numbers.Number):
            weight = weight * torch.ones(y.shape[:-1], device=y.device)
        nb = y.shape[0]
        alpha = torch.ones(nb, device=y.device)
        j = 0
        while True:
            idx_all = torch.arange(nb, device=y.device)
            x = project(y)
            c, c_error_mean, c_error_max = self.constraint_violation(x)
            if j == 0:
                reg = c_error_mean
                reg2 = (c * c).mean()
                cm_norm_init = c.abs().max()
            # cm = c.abs().max(dim=2)[0].max(dim=2)[0]
            cm_norm = c.abs().amax(dim=(1, 2))
            M = cm_norm > self.tolerance
            idx = idx_all[M]
            if len(idx) == 0:
                break
            else:
                c = self._constraint(x)
                cm = c.abs().mean(dim=(1, 2))
            if debug_idx is not None:
                self.debug(x, c, extra=j, idx=debug_idx)
            dx = self._jacobian_transpose_times_constraint(x[idx], c[idx])
            dx = weight[idx].view(dx.shape) * dx
            dy = uplift(dx)
            lsiter = torch.zeros(len(idx), device=y.device)
            while True:
                y_try = y[idx] - alpha[idx, None, None] * dy
                x_try = project(y_try)
                c_try = self._constraint(x_try)
                cm_try = c_try.abs().mean(dim=(1, 2))
                M_try = cm_try <= cm[idx]
                if M_try.all():
                    break
                idx_sel = idx[~M_try]
                alpha[idx_sel] = alpha[idx_sel] / 2
                lsiter[~M_try] = lsiter[~M_try] + 1
                if lsiter.max() > 20:
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
            if alpha.min() < self.tolerance:
                print("Alpha too small")
                break
            if j > self.max_iter:
                print("Projection failed")
                break
        # print(f"projection {j} steps. Max violation: {cm_norm_init.item():2.2e} -> {cm_norm.max().item():2.2e}")
        # print(alpha.min().item())
        return y, reg, reg2

    def constraint_violation(self, x):
        """
        Computes the constraint violation.
        """
        try:
            c = self._constraint_violation(x, rescale=True)
        except:
            c = self._constraint(x, rescale=True)
        cabs = torch.abs(c)
        c_error_mean = torch.mean(cabs)
        c_error_max = torch.max(cabs)
        return c, c_error_mean, c_error_max

    def __call__(self, y, project=nn.Identity(), uplift=nn.Identity(), weight=1, use_batch=True, K=None):
        if use_batch and self.minimizer == 'gradient_descent':
            y, reg, reg2 = self.gradient_descent_batch(y, project, uplift, weight)
        else:
            nb = y.shape[0]
            y_new = []
            reg = 0
            reg2 = 0
            for i in range(nb):
                if self.minimizer == 'gradient_descent':
                    tmp, regi, regi2, problems = self.gradient_descent(y[i:i + 1], project, uplift, K, weight[i:i + 1])
                    if problems:
                        tmp, regi, regi2, problems = self.gradient_descent(y[i:i + 1], project, uplift, K, weight[i:i + 1])
                elif self.minimizer == 'cg':
                    tmp, regi, regi2 = self.conjugate_gradient(y[i:i + 1], K=K, weight=weight[i:i + 1])
                else:
                    raise NotImplementedError("the mode you have selected is not implemented yet")
                y_new.append(tmp)
                reg = reg + regi
                reg2 = reg2 + regi2
            y = torch.cat(y_new, dim=0)
        return y, reg, reg2


class MultiBodyPendulum(ConstraintTemplate):
    """
    This constraint function applies multi-body pendulum constraints.
    Expected input is x of shape [batch_size,npendulums,ndims].

    Input:
        l: A torch tensor with the length of the different pendulums, can also be a single number if all pendulums have the same length.
        position_idx: gives the indices for x and y coordinates in ndims.

    Note that this class is quite complicated and has both a _constraint and _constraint_violation function which is typically not needed.
    The reason both of these exist is that we use _constraint for the actual constraints used by the backpropagation, hence these constraints should be convex if possible and with vanishing gradients as the constraints goes to zero.
    _constraint_violation is used to calculate the actual constraint violation in sensible units for reporting purposes.
    """

    def __init__(self, position_idx=(0, 1), velocity_idx=(2, 3), **kwargs):
        super(MultiBodyPendulum, self).__init__(**kwargs)
        self.l = 1
        self.position_idx = position_idx
        self.velocity_idx = velocity_idx
        return

    def _delta_r(self, r):
        """
        Computes a vector from each pendulum to the next, including origo.
        """
        dr_0 = r[:, 0]
        dr_i = r[:, 1:] - r[:, :-1]
        dr = torch.cat((dr_0[:, None], dr_i), dim=1)
        return dr

    def _extract_positions(self, x):
        """
        Extracts positions, r, from x
        """
        r = x[:, :, self.position_idx]
        return r

    def _extract_velocity(self, x):
        """
        Extracts velocities, v, from x
        """
        v = x[:, :, self.velocity_idx]
        return v

    def _insert_positions(self, r, x_template):
        """
        Inserts, r, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:, :, self.position_idx] = r
        return x

    def _insert_velocities(self, v, x_template):
        """
        Inserts, v, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:, :, self.velocity_idx] = v
        return x

    def _constraint_base(self, x, rescale=False):
        """
        Computes the constraints.

        For a n multi-body pendulum the constraint can be given as:
        c_i = (r_i - r_{i-1})**2 - l_i**2,   i=1,n

        Note that this should not be used when computing the constraint violation since that should be done with the unsquared norm.
        """
        r = self._extract_positions(x)
        dr = self._delta_r(r)
        if rescale:
            dr = dr * self.scale
        # drnorm = torch.norm(dr, dim=-1)
        dr2 = (dr * dr).sum(dim=-1)
        c = dr2 - self.l * self.l
        return c[:, :, None]

    def _constraint(self, x, rescale=False):
        """
        Computes the constraints, and potentially the second order constraints and combines them.
        """
        c1 = self._constraint_base(x, rescale=rescale)
        if self.include_second_order_constraints:
            v = self._extract_velocity(x)
            r = self._extract_positions(x)
            c2 = self._second_order_constraint(self._constraint_base, r, v)
            c = torch.cat((c1, c2), dim=-1)
        else:
            c = c1
        return c

    def _constraint_violation_base(self, x, rescale=False):
        """
        Computes the constraints.

        For an n multi-body pendulum the constraint can be given as:
        c_i = ||r_i - r_{i-1}||_2 - l_i,   i=1,n
        """
        r = self._extract_positions(x)
        dr = self._delta_r(r)
        if rescale:
            dr = dr * self.scale
        drnorm = torch.norm(dr, dim=-1)
        c = drnorm - self.l
        return c[:, :, None]

    def _constraint_violation(self, x, rescale=False):
        """
        Computes the constraints violation, and potentially the second order constraints and combines them.
        """
        c1 = self._constraint_violation_base(x, rescale=rescale)
        if self.include_second_order_constraints:
            v = self._extract_velocity(x)
            r = self._extract_positions(x)
            c2 = self._second_order_constraint(self._constraint_violation_base, r, v)
            c = torch.cat((c1, c2), dim=-1)
        else:
            c = c1
        return c


class Water(ConstraintTemplate):
    """
    This constraint function applies constraints to water molecules.
    Expected input is x of shape [batch_size,nwater,ndims].

    Note that technically this keeps applying the constraint to water molecules that falls within the acceptable limit
     until all water molecules in the batch falls within that limit, so a more correct approach would be to put each molecule in the batch_size
     and have a shape that looks like [nwater*batch_size,1,ndims].

    Input:
        l: A torch tensor with the binding lengths of the different bonds, can also be a single number if all bonds have the same length.
        position_idx: gives the indices for x,y,z coordinates in ndims for the different particles.
    """

    def __init__(self, length, position_idx=(0, 1, 2, 3, 4, 5, 6, 7, 8), velocity_idx=(9, 10, 11, 12, 13, 14, 15, 16, 17), **kwargs):
        super(Water, self).__init__(**kwargs)
        if not isinstance(length, torch.Tensor):
            length = torch.tensor(length) / self.scale
        self.register_buffer('l', length)
        self.position_idx = position_idx
        self.velocity_idx = velocity_idx
        return

    def _extract_positions(self, x):
        """
        Extracts positions, r, from x
        """
        r = x[:, :, self.position_idx]
        r = r.view(r.shape[0], r.shape[1], -1, 3)
        return r

    def _insert_positions(self, r, x_template):
        """
        Inserts, r, back into a zero torch tensor similar to x_template at the appropriate spot.
        """
        x = torch.zeros_like(x_template)
        x[:, :, self.position_idx] = r.view(r.shape[0], r.shape[1], -1)
        return x

    def _delta_r(self, r):
        """
        computes the internal distances in each water molecule
        """
        dr1 = r[:, :, 0] - r[:, :, 1]
        dr2 = r[:, :, 1] - r[:, :, 2]
        dr3 = r[:, :, 2] - r[:, :, 0]
        dr = torch.cat((dr1[:, :, None, :], dr2[:, :, None, :], dr3[:, :, None, :]), dim=2)
        return dr

    def _constraint(self, x, extract_position=True, rescale=False):
        """
        Computes the constraints.

        For a water molecule the constraint can be given as:
        c_1 = |r_1 - r_2| - l_1
        c_2 = |r_2 - r_3| - l_2
        c_3 = |r_3 - r_1| - l_3
        """
        if extract_position:
            r = self._extract_positions(x)
        else:
            r = x
        dr = self._delta_r(r)
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

    alpha_rad = torch.acos((l0 ** 2 + l2 ** 2 - l1 ** 2) / (2 * l0 * l2))
    alpha_degree = alpha_rad / torch.pi * 180

    l = torch.tensor([l0, l1, l2])
    beta = (180 - alpha_degree) / 2  # degrees
    b = torch.tensor(beta / 180 * torch.pi)

    r1 = torch.tensor([[0.0, 0.0, 0.0], [torch.cos(b) * l0, torch.sin(b) * l0, 0.0], [-torch.cos(b) * l0, torch.sin(b) * l0, 0.0]])
    r = r1[None, None, :]

    con_fnc = Water(l, tolerance=1e-2, max_iter=100)
    c = con_fnc.constraint(r, extract_position=False)
    assert c.allclose(torch.tensor(0.0))
