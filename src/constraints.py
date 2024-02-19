import numbers
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F


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
    elif name == 'imagedenoising':
        con_fnc = ImageDenoising(**c_dict)
    elif name == None:
        con_fnc = None
    else:
        raise NotImplementedError(f"The constraint {name} has not been implemented.")
    return con_fnc


class ConstraintTemplate(ABC, nn.Module):
    """
    This is the template class for all constraints.
    Each constraint should have this class as their parent.

    Technically the constraints can have whatever shape you want, but if the constraints should be usable with the minimization functions already built they need to follow the following conventions.
    The expected input shape of a parameter being minimized is:
    [nb,nc,ndim]
    where
    nb is the number of batches (so separate samples that have no constraint connection) Note that this does not have to be the same as regular batch_size
    nc is the number of coupled constraints in a batch
    ndim is the number of dimensions for each constraint


    Input:
        include_second_order_constraints: If True it enables second order constraints. That is: \dot{c} = C(x) \cdot \dot{x} = 0,
                                          where C(x) = \frac{\partial c}{\partial x}, and c is the first order constraints
        max_penalty_change: This is the fractional maximum change allowed when using penalty or stabilization as it is also commonly called: dy = J^T c.
    """

    def __init__(self, include_second_order_constraints=False, max_penalty_change=0.1):
        super(ConstraintTemplate, self).__init__()
        self.include_second_order_constraints = include_second_order_constraints
        self.max_penalty_change =max_penalty_change
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

    def constraint_penalty(self, y, project_fnc=nn.Identity(), uplift_fnc=nn.Identity(), weight=1):
        """
        Calculates constraint stabilization as dy = J^T c
        Note that we have added a clamp, which prevents this penalty from creating changes larger than a certain fraction of the value of x.
            Without this clamp, the penalty changes can in rare cases be unstable.
        """
        x = project_fnc(y)
        c = self._constraint(x)
        dx = self._jacobian_transpose_times_constraint(x, c)
        if weight == 1:
            pass
        else:
            dx = weight.view(dx.shape) * dx
        clampval = x.abs() * self.max_penalty_change
        dx = torch.clamp(dx, min=-clampval, max=clampval)
        dy = uplift_fnc(dx)
        return dy


    def constraint_violation(self, y, project_fnc=nn.Identity()):
        """
        Computes the constraint violation.
        """
        x = project_fnc(y)
        try:
            c = self._constraint_violation(x, rescale=True)
        except:
            c = self._constraint(x, rescale=True)
        cabs = torch.abs(c)
        c_error_mean = torch.mean(cabs)
        c_error_max = torch.max(cabs)
        return c, c_error_mean, c_error_max

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

    def __init__(self, position_idx=(0, 1), velocity_idx=(2, 3), scale=1, **kwargs):
        super(MultiBodyPendulum, self).__init__(**kwargs)
        self.scale = scale
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

    def __init__(self, length, position_idx=(0, 1, 2, 3, 4, 5, 6, 7, 8), velocity_idx=(9, 10, 11, 12, 13, 14, 15, 16, 17), scale=1, **kwargs):
        super(Water, self).__init__(**kwargs)
        self.scale = scale
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
        c = drnorm - l
        return c




class ImageDenoising(ConstraintTemplate):
    """
    This constraint function applies constraints that ensure that a pair of images are divergence free.
    Expected input is x of shape [batch_size,2,npixels,npixels].
    """

    def __init__(self, scale=1, **kwargs):
        super(ImageDenoising, self).__init__(**kwargs)
        self.scale = scale
        Dx = torch.tensor([[-1.0, 1.0], [-1, 1]]).reshape(1, 1, 2, 2)
        Dy = torch.tensor([[-1.0, -1.0], [1, 1]]).reshape(1, 1, 2, 2)
        D = torch.cat((Dx, Dy), dim=0)
        # self.D = D
        self.register_buffer('D', D)

        return

    def _constraint(self, x, rescale=False):
        """
        Computes the constraints.

        """
        c = F.conv2d(x, self.D, groups=2)
        c = c.sum(dim=(1))
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
