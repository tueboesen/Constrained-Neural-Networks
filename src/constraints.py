import inspect

import torch
import torch.nn as nn
from torch.autograd import grad


def load_constraints(con,con_type,project_fnc=None,uplift_fnc=None,con_variables=None,rscale=1,vscale=1,pos_only=False,debug=False):
    """
    This is a wrapper function for loading constraints.

    masses=None,R=None,V=None,z=None,rscale=1,vscale=1,energy_predictor=None
    """
    if con == 'chain':
        d0 = con_variables['d0']
        con_fnc = torch.nn.Sequential(PointChain(d0,con_type,project_fnc,uplift_fnc, pos_only=pos_only,debug=debug))
    elif con == 'triangle':
        r0 = con_variables['r0']
        r1 = con_variables['r1']
        r2 = con_variables['r2']
        PTP = PointToPoint(r0/rscale,con_type,project=project_fnc,uplift=uplift_fnc,pos_only=pos_only,debug=debug)
        PTSSI = PointToSphereSphereIntersection(r1/rscale,r2/rscale,con_type=con_type,project=project_fnc,uplift=uplift_fnc,pos_only=pos_only,debug=debug)
        con_fnc = torch.nn.Sequential(PTP,PTSSI)
    elif con == 'chaintriangle':
        d0 = con_variables['d0']
        r0 = con_variables['r0']
        r1 = con_variables['r1']
        r2 = con_variables['r2']
        PC = PointChain(d0, con_type, project_fnc, uplift_fnc, pos_only=pos_only, debug=debug)
        PTP = PointToPoint(r0/rscale,con_type,project=project_fnc,uplift=uplift_fnc,pos_only=pos_only,debug=debug)
        PTSSI = PointToSphereSphereIntersection(r1/rscale,r2/rscale,con_type=con_type,project=project_fnc,uplift=uplift_fnc,pos_only=pos_only,debug=debug)
        con_fnc = torch.nn.Sequential(PC,PTP,PTSSI)
    elif con == 'P': # Momentum constraints
        mass = con_variables['mass']
        M = MomentumConstraints(project_fnc, uplift_fnc, mass)
        con_fnc = torch.nn.Sequential(M)
    elif con == 'EP': # Joint energy momentum constraints. Note that the energy constraints needs a pretrained neural network for predicting the force/potential energy.
        energy_predictor = con_variables['energy_predictor']
        mass = con_variables['mass']
        # force_predictor = generate_FE_network(natoms=z.shape[1])
        # force_predictor.load_state_dict(torch.load(energy_predictor, map_location=torch.device('cpu')))
        # force_predictor.eval()
        EM = EnergyMomentumConstraints(project_fnc,uplift_fnc, energy_predictor, mass, rescale_r=rscale, rescale_v=vscale)
        con_fnc = torch.nn.Sequential(EM)
        # constraints[0].fix_reference_energy(R,V,z)
    elif con == '':
        con_fnc = None
    else:
        raise NotImplementedError("The constraint chosen has not been implemented.")
    return con_fnc

def load_constraint_parameters(con,con_type,data_type,con_data=''):
    """
    Loads the constraints parameters, either from a data file if one is provided, or from the function constraint_hyperparameters.
    """
    if con_data == '':
        cv = constraint_hyperparameters(con,con_type,data_type)
    else:
        cv = torch.load(con_data)
    return cv

def constraint_hyperparameters(con,con_type,data_type):
    """
    Here we store some of the simpler constraint variables needed. More complicated constraints should be saved to a file and loaded instead.
    """
    cv = {}
    if con == 'chain':
        if data_type == 'proteins':
            cv['d0'] = 3.8 #Angstrom, ensure that your units are correct.
        else:
            NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con,data_type,inspect.currentframe().f_code.co_name))
    elif con == 'triangle':
        if data_type == 'water':
            cv['r0'] = 0.957
            cv['r1'] = 0.957
            cv['r2'] = 1.513
        else:
            NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con,data_type,inspect.currentframe().f_code.co_name))
    else:
        NotImplementedError("The combination of constraints={:} and data_type={:} has not been implemented in function {:}".format(con, data_type, inspect.currentframe().f_code.co_name))

    return cv



class ConstraintTemplate(nn.Module):
    """
    This is the template class for all constraints.
    Each constraint should have this class as their parent.

    data should be a dictionary containing the variables that needs to be constrained, but can also contain other things, which will then be left untouched.

    The convention used in this project is the following:
    high dimensional data is named 'y'
    low dimensional data is named 'x'
    regularization constraints will be saved in 'c'
    """

    def __init__(self, con_type):
        super(ConstraintTemplate, self).__init__()
        self.con_type = con_type
        return

    def constrain_high_dimension(self, data):
        raise NotImplementedError(
            "Constraints in high dimension have not been implemented for {:}".format(self._get_name()))

    def constrain_low_dimension(self, data):
        raise NotImplementedError(
            "Constraints in low dimension have not been implemented for {:}".format(self._get_name()))

    def constrain_regularization(self, data):
        raise NotImplementedError(
            "Regularization constraints have not been implemented for {:}".format(self._get_name()))

    def forward(self, data):
        if self.con_type == 'high':
            data = self.constrain_high_dimension(data)
        elif self.con_type == 'low':
            data = self.constrain_low_dimension(data)
        elif self.con_type == 'reg':
            data = self.constrain_regularization(data)
        else:
            raise NotImplementedError(
                "The constrain type: {:}, you have selected is not implemented for {:}".format(self.con_type,self._get_name()))
        return data

class PointToPoint(ConstraintTemplate):
    """
    This is a point to point constraint, which constraints the distance between 2 points, by moving the second point to the distance required.
    d is the distance the points get constrained to, it can either be a vector or a scalar (vectors are used when different distances are needed for different types of particles)
    constrain_type can either be 'high', 'low', or 'reg'
    project and uplift are functions that performs a projection and uplifting operations, they are only needed for high dimensional constraining, and can otherwise be omitted.
    pos_only can be used if the data only contains positions rather than positions and velocities.

    y is the high dimensional variable, ordered as [nparticles,latent_dim]
    x is the low dimensional variable, ordered as [nparticles,9/18 dim] (depending on whether pos_only is true or not. If pos_only=False, then the ordering should be r,v)

    """
    def __init__(self,r0,con_type,project=None,uplift=None,pos_only=False,debug=False):
        super(PointToPoint, self).__init__(con_type)
        self.r0 = r0
        self.project = project
        self.uplift = uplift
        self.pos_only = pos_only
        self.debug = debug
        if con_type == 'high':
            if project is None:
                raise ValueError("For high dimensional constraints a projection function is needed.")
            if uplift is None:
                raise ValueError("For high dimensional constraints an uplifting function is needed.")
        return

    def constraint(self,p1,p2,z):
        if self.r0.ndim==0:
            r0 = self.r0
        else:
            r0 = (self.r0[z]).to(device=p1.device,dtype=p1.dtype)
        d = p2 - p1
        lam = d * (1 - (r0 / d.norm(dim=-1).unsqueeze(-1)))
        return lam

    def constrain_high_dimension(self,data):
        batch = data['batch']
        z = data['z']
        y = data['y']
        x = self.project(y)
        if self.pos_only:
            ndim = x.shape[-1]
        else:
            ndim = x.shape[-1] // 2
        lam_x = torch.zeros_like(x)
        nb = batch.max() + 1
        r = x[:, 0:ndim].view(nb, -1, ndim)
        z2 = z.view(nb,-1,1)
        lam_p2 = self.constraint(r[:, :, 0:3], r[:, :, 3:6],z2)
        lam_x[:, 3:6] = lam_p2.view(-1,3)
        if self.pos_only:
            lam_y = self.uplift(lam_x.view(-1, ndim))
        else:
            lam_y = self.uplift(lam_x.view(-1, 2 * ndim))
        y = y - lam_y
        if self.debug:
            x = self.project(y)
            r = x[:, 0:ndim].view(nb, -1, ndim)
            lam_p2_after = self.constraint(r[:, :, 0:3], r[:, :, 3:6],z2)
            cnorm = torch.mean(torch.sum(lam_p2 ** 2, dim=-1))
            cnorm_after = torch.mean(torch.sum(lam_p2_after ** 2, dim=-1))
            print(f"{self._get_name()} constraint c: {cnorm:2.4f} -> {cnorm_after:2.4f}")
        return {'y': y, 'batch': batch, 'z': z}

    def constrain_low_dimension(self, data):
        batch = data['batch']
        nb = batch.max() + 1
        x = data['x']
        z = data['z']
        if self.pos_only:
            ndim = x.shape[-1]
        else:
            ndim = x.shape[-1] // 2
        r = x[:, 0:ndim].view(nb, -1, ndim)
        z2 = z.view(nb,-1,1)
        # v = x[:, ndim:].view(nb, -1, ndim)
        lam_p2 = self.constraint(r[:, :, 0:3], r[:, :, 3:6],z2)
        x[:,3:6] = x[:,3:6] - lam_p2.view(-1,3)
        if self.debug:
            r = x[:, 0:ndim].view(nb, -1, ndim)
            lam_p2_after = self.constraint(r[:, :, 0:3], r[:, :, 3:6],z2)
            cnorm = torch.mean(torch.sum(lam_p2 ** 2, dim=-1))
            cnorm_after = torch.mean(torch.sum(lam_p2_after ** 2, dim=-1))
            print(f"{self._get_name()} constraint c: {cnorm:2.8f} -> {cnorm_after:2.8f}")
        return {'x': x, 'batch': batch,'z':z}

    def constrain_regularization(self, data):
        x = data['x']
        z = data['z']
        if 'c' in data:
            c = data['c']
        else:
            c = 0
        r = x.view(1, -1, 9)
        z2 = z.view(1, -1, 1)
        lam_p2 = self.constraint(r[:, :, 0:3], r[:, :, 3:6], z2)
        c_new = torch.sum(lam_p2 ** 2)/100
        return {'x': x, 'z': z, 'c': c + c_new}

class PointToSphereSphereIntersection(ConstraintTemplate):
    """
    This is a point to sphere sphere intersection constraint, which is basically just a fancy way of saying that we have 2 fixed points, and now wish to fixate a third point such that it is distance r1 from point 1, and r2 from point 2, while moving the point as little as possible.
    r1,r2 are the distances the points get constrained to, they can either be vectors or scalars (vectors are used when different distances are needed for different types of particles)
    constrain_type can either be 'high', 'low', or 'reg'
    project and uplift are functions that performs a projection and uplifting operations, they are only needed for high dimensional constraining, and can otherwise be omitted.
    pos_only can be used if the data only contains positions rather than positions and velocities.

    y is the high dimensional variable, ordered as [nparticles,latent_dim]
    x is the low dimensional variable, ordered as [nparticles,9/18 dim] (depending on whether pos_only is true or not. If pos_only=False, then the ordering should be r,v)

    """
    def __init__(self,r1,r2,con_type,project=None,uplift=None,pos_only=False,debug=False):
        super(PointToSphereSphereIntersection, self).__init__(con_type)
        self.r1 = r1
        self.r2 = r2
        self.project = project
        self.uplift = uplift
        self.pos_only = pos_only
        self.debug = debug
        if con_type == 'high':
            if project is None:
                raise ValueError("For high dimensional constraints a projection function is needed.")
            if uplift is None:
                raise ValueError("For high dimensional constraints an uplifting function is needed.")
        return

    def constraint(self,p1,p2,p3,z):
        if self.r1.ndim == 0:
            r1 = self.r1
            r2 = self.r2
        else:
            r1 = (self.r1[z]).to(device=p1.device,dtype=p1.dtype)
            r2 = (self.r2[z]).to(device=p1.device,dtype=p1.dtype)
        d = p2 - p1
        dn = d.norm(dim=-1)
        a = 1 / (2 * dn) * torch.sqrt(4 * dn ** 2 * r2 ** 2 - (dn ** 2 - r1 ** 2 + r2 ** 2)**2)

        cn = (dn ** 2 - r2 ** 2 + r1 ** 2) / (2 * dn)
        c = cn[:,:,None] / dn[:,:,None] * d + p1
        n = d / dn.unsqueeze(-1)

        q = p3 - c - (torch.sum(n*(p3 - c),dim=-1,keepdim=True) * n)
        K = c + a.unsqueeze(-1) * q / q.norm(dim=-1).unsqueeze(-1)

        lam_p3 = -(K - p3)
        return lam_p3

    def constrain_low_dimension(self,data):
        x = data['x']
        batch = data['batch']
        z = data['z']
        if self.pos_only:
            ndim = x.shape[-1]
        else:
            ndim = x.shape[-1] // 2
        nvec = ndim // 3
        lam_x = torch.zeros_like(x)
        nb = batch.max() + 1
        r = x[:, 0:ndim].view(nb, -1, ndim)
        z2 = z.view(nb,-1)
        # v = x[:, ndim:].view(nb, -1, ndim)

        lam_p3 = self.constraint(r[:,:,0:3],r[:,:,3:6],r[:,:,6:9],z2)
        x[:,6:9] = x[:,6:9] - lam_p3.view(-1,3)
        if self.debug:
            r = x[:, 0:ndim].view(nb, -1, ndim)
            lam_p3_after = self.constraint(r[:, :, 0:3], r[:, :, 3:6], r[:, :, 6:9],z2)
            cnorm = torch.mean(torch.sum(lam_p3 ** 2, dim=-1))
            cnorm_after = torch.mean(torch.sum(lam_p3_after ** 2, dim=-1))
            print(f"{self._get_name()} constraint c: {cnorm:2.8f} -> {cnorm_after:2.8f}")
        return {'x':x,'batch':batch,'z':z}

    def constrain_high_dimension(self,data):
        y = data['y']
        z = data['z']
        batch = data['batch']
        # for j in range(n):
        x = self.project(y)
        if self.pos_only:
            ndim = x.shape[-1]
        else:
            ndim = x.shape[-1]//2
        nvec = ndim // 3
        lam_x = torch.zeros_like(x)
        nb = batch.max() + 1
        r = x[:, 0:ndim].view(nb, -1, ndim)
        z2 = z.view(nb,-1)
        # v = x[:, ndim:].view(nb, -1, ndim)

        lam_p3 = self.constraint(r[:,:,0:3],r[:,:,3:6],r[:,:,6:9],z2)
        lam_x[:,6:9] = lam_p3.view(-1,3)
        if self.pos_only:
            lam_y = self.uplift(lam_x.view(-1, ndim))
        else:
            lam_y = self.uplift(lam_x.view(-1, 2 * ndim))
        y = y - lam_y
        if self.debug:
            x = self.project(y)
            r = x[:, 0:ndim].view(nb, -1, ndim)
            v = x[:, ndim:].view(nb, -1, ndim)
            lam_p3_after = self.constraint(r[:, :, 0:3], r[:, :, 3:6], r[:, :, 6:9],z2)
            cnorm = torch.mean(torch.sum(lam_p3 ** 2, dim=-1))
            cnorm_after = torch.mean(torch.sum(lam_p3_after ** 2, dim=-1))
            print(f"{self._get_name()} constraint c: {cnorm:2.8f} -> {cnorm_after:2.8f}")
        return {'y':y,'batch':batch, 'z':z}

    def constrain_regularization(self, data):
        x = data['x']
        z = data['z']
        if 'c' in data:
            c = data['c']
        else:
            c = 0
        r = x.view(1, -1, 9)
        z2 = z.view(1, -1)
        p1 = r[:, :, 0:3]
        p2 = r[:, :, 3:6]
        p3 = r[:, :, 6:9]
        r1 = (self.r1[z2]).to(device=x.device,dtype=x.dtype)
        r2 = (self.r2[z2]).to(device=x.device,dtype=x.dtype)

        d23 = (p3 - p2).norm(dim=-1)
        d13 = (p3 - p1).norm(dim=-1)

        c23 = torch.sum((d23 - r2) ** 2)
        c13 = torch.sum((d13 - r1) ** 2)
        c_new = (c23 + c13)/100
        return {'x': x, 'z': z, 'c': c + c_new}


class PointChain(ConstraintTemplate):
    """
    This is a PointChain constraint, which ensures that neighbouring points all have a fixed distance, d0, between them.

    d0 is the distances the points get constrained to, it can either be a vector or a scalar (vectors are used when different distances are needed for different types of particles)
    constrain_type can either be 'high', 'low', or 'reg'
    project and uplift are functions that performs a projection and uplifting operations, they are only needed for high dimensional constraining, and can otherwise be omitted.
    pos_only can be used if the data only contains positions rather than positions and velocities.

    niter and converged_acc are used to determine how many iterations are used to try to enforce the constraint, n_iter is the maxmimum number of iterations to use, while converged_acc is the accuracy at which it will stop iterating.

    y is the high dimensional variable, ordered as [nparticles,latent_dim]
    x is the low dimensional variable, ordered as [nparticles,9/18 dim] (depending on whether pos_only is true or not. If pos_only=False, then the ordering should be r,v)

    """
    def __init__(self,d0,con_type,project=None,uplift=None,pos_only=False,debug=False, niter=10, converged_acc=1e-9):
        super(PointChain, self).__init__(con_type)
        self.d0 = d0
        self.project = project
        self.uplift = uplift
        self.pos_only = pos_only
        self.debug = debug
        self.niter = niter
        self.converged_acc = converged_acc
        if con_type == 'high':
            if project is None:
                raise ValueError("For high dimensional constraints a projection function is needed.")
            if uplift is None:
                raise ValueError("For high dimensional constraints an uplifting function is needed.")
        return

    def diff(self,x):
        return x[:,1:] - x[:,:-1]

    def diffT(self,dx):
        x = dx[:,:-1] - dx[:,1:]
        x0 = -dx[:,:1]
        x1 = dx[:,-1:]
        X = torch.cat([x0,x,x1],dim=1)
        return X

    def constraint(self,x,z):
        if self.d0.ndim==0:
            d = self.d0
        else:
            d = (self.d0[z[:, :-1], z[:, 1:]]).to(device=x.device,dtype=x.dtype)
        e = torch.ones((3,1),device=x.device)
        dx = self.diff(x)
        c = (dx**2)@e - d**2
        return c

    def dConstraintT2(self,c, X):
         dX = self.diff(X)
         e = torch.ones(1, 3, device=X.device)
         C = (c @ e) * dX
         C2 = self.diffT(C)
         return 2 * C2

    def constrain_high_dimension(self, data):
        """
            fragmentid is used when the input consists of multiple fragments, in that case the constraints are only applied piecewise to points sharing the same fragmentid. If all the points are in the same fragment the variable can just be ignored.
        """
        y = data['y']
        batch = data['batch']
        z = data['z']
        if 'fragmentid' in data:
            fragid = data['fragmentid']
            fragid_unique = torch.unique(fragid)
        else:
            fragid = -1
            fragid_unique = [-1]
        for j in range(self.niter):
            x = self.project(y)
            if self.pos_only:
                ndim = x.shape[-1]
            else:
                ndim = x.shape[-1] // 2
            nvec = ndim // 3
            lam_x = torch.zeros_like(x)
            nb = batch.max() + 1
            r = x[:, 0:ndim].view(nb, -1, ndim)
            z2 = z.view(nb,-1,1)
            cnorm = 0
            for fragi in fragid_unique:
                if fragi != -1:
                    idx = fragid == fragi
                else:
                    idx = r[0,:,0] == r[0,:,0]
                ri = r[:,idx,:]
                ria = ri[:, :,:3]
                ci = self.constraint(ria,z2)
                # lam_xia = self.dConstraintT(ci, ria)
                lam_xia = self.dConstraintT2(ci, ria)
                lam_xi = lam_xia.repeat(1,1,nvec)
                lam_x[idx,:ndim] = lam_xi
                cnormi = torch.sum(ci**2)
                cnorm = cnorm + cnormi
            if self.pos_only:
                lam_y = self.uplift(lam_x.view(-1, ndim))
            else:
                lam_y = self.uplift(lam_x.view(-1, 2 * ndim))
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:ndim].view(nb, -1, ndim)
                    ctry_norm = 0
                    for fragi in fragid_unique:
                        if fragi != -1:
                            idx = fragid == fragi
                        else:
                            idx = r[0, :, 0] == r[0, :, 0]
                        ri = r[:,idx,:]
                        ria = ri[:, :, :3]
                        ctry = self.constraint(ria,z2)
                        ctry_norm = ctry_norm + torch.sum(ctry**2)
                    if ctry_norm < cnorm:
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0 and ctry_norm > self.converged_acc:
                    alpha = alpha * 1.5
            y = y - alpha * lam_y
            if self.debug:
                print(f"{self._get_name()} constraints {j} c: {cnorm.detach().cpu():2.4f} -> {ctry_norm.detach().cpu():2.4f}   ")
            if ctry_norm < self.converged_acc:
                break
        return {'y':y,'batch':batch, 'z': z}


    def constrain_low_dimension(self, data):
        x = data['x']
        batch = data['batch']
        z = data['z']
        if 'fragmentid' in data:
            fragid = data['fragmentid']
            fragid_unique = torch.unique(fragid)
        else:
            fragid = -1
            fragid_unique = [-1]
        for j in range(self.niter):
            if self.pos_only:
                ndim = x.shape[-1]
            else:
                ndim = x.shape[-1] // 2
            nvec = ndim // 3
            lam_x = torch.zeros_like(x)
            nb = batch.max() + 1
            r = x[:, 0:ndim].view(nb, -1, ndim)
            z2 = z.view(nb,-1,1)
            cnorm = 0
            for fragi in fragid_unique:
                if fragi != -1:
                    idx = fragid == fragi
                else:
                    idx = r[0,:,0] == r[0,:,0]
                ri = r[:,idx,:]
                ria = ri[:, :,:3]
                ci = self.constraint(ria,z2)
                # lam_xia = self.dConstraintT(ci, ria)
                lam_xia = self.dConstraintT2(ci, ria)

                # lam_p3 = self.constraint(r[:, :, 0:3], r[:, :, 3:6], r[:, :, 6:9], z2)
                # x[:, 6:9] = x[:, 6:9] - lam_p3.view(-1, 3)

                lam_xi = lam_xia.repeat(1,1,nvec)
                lam_x[idx,:ndim] = lam_xi
                cnormi = torch.sum(ci**2)
                cnorm = cnorm + cnormi
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_x.norm()
                lsiter = 0
                while True:
                    xtry = x - alpha * lam_x
                    rtry = xtry[:, 0:ndim].view(nb, -1, ndim)
                    ctry_norm = 0
                    for fragi in fragid_unique:
                        if fragi != -1:
                            idx = fragid == fragi
                        else:
                            idx = r[0, :, 0] == r[0, :, 0]
                        ri = rtry[:,idx,:]
                        ria = ri[:, :, :3]
                        ctry = self.constraint(ria,z2)
                        ctry_norm = ctry_norm + torch.sum(ctry**2)
                    if ctry_norm < cnorm:
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0 and ctry_norm > self.converged_acc:
                    alpha = alpha * 1.5
            x = x - alpha * lam_x
            if self.debug:
                print(f"{self._get_name()} constraints {j} c: {cnorm.detach().cpu():2.4f} -> {ctry_norm.detach().cpu():2.4f}   ")
            if ctry_norm < self.converged_acc:
                break
        return {'x':x,'batch':batch, 'z': z}

    def constrain_regularization(self, data):
        x = data['x']
        z = data['z']
        if 'c' in data:
            c = data['c']
        else:
            c = 0
        z2 = z.view(1, -1, 1)
        r = x[:, :3].view(1, -1, 3)
        cc = self.constraint(r, z2)
        c_new = torch.sum(torch.abs(cc))/100
        return {'x': x, 'z': z, 'c': c + c_new}


class MomentumConstraints(nn.Module):
    def __init__(self,project,uplift,m):
        super(MomentumConstraints, self).__init__()
        self.register_buffer("m", m[:,None])
        self.project = project
        self.uplift = uplift
        return


    def constraint(self,v):
        m = self.m
        P = v.transpose(1,2) @ m
        return P

    def dConstraintT(self,v,c):
        m = self.m
        e3 = torch.ones((3,1),dtype=v.dtype,device=v.device)
        en = torch.ones((m.shape[0],1),dtype=v.dtype,device=v.device)
        jv = m @ e3.T * (en @ c.transpose(1,2))
        jr = torch.zeros_like(jv)
        j = torch.cat([jr,jv],dim=-1)
        return j

    def forward(self, data, n=10, debug=False, converged=1e-4):
        y = data['y']
        batch = data['batch']

        for j in range(n):
            x = self.project(y)
            nb = batch.max()+1
            r = x[:, 0:3].view(nb,-1,3)
            v = x[:, 3:].view(nb,-1,3)
            c = self.constraint(v)
            lam_x = self.dConstraintT(v,c)
            cnorm = torch.norm(c)
            lam_y = self.uplift(lam_x.view(-1,6))
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:3].view(nb, -1, 3)
                    v = x[:, 3:].view(nb, -1, 3)
                    ctry = self.constraint(v)
                    ctry_norm = torch.norm(ctry)
                    if ctry_norm < cnorm:
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0 and ctry_norm > converged:
                    alpha = alpha * 1.5
            y = y - alpha * lam_y
            if debug:
                print(f"{self._get_name()} constraints {j} c: {cnorm.detach().cpu():2.4f} -> {ctry_norm.detach().cpu():2.4f}   ")
            if ctry_norm < converged:
                break
        return {'y':y,'batch':batch}



class EnergyMomentumConstraints(nn.Module):
    def __init__(self,project, uplift, potential_energy_predictor,m,rescale_r=1,rescale_v=1):
        super(EnergyMomentumConstraints, self).__init__()
        self.register_buffer("m", m[None,:,None])
        self.project = project
        self.uplift = uplift
        self.pep = potential_energy_predictor
        self.pep.eval()
        self.rescale_r = rescale_r
        self.rescale_v = rescale_v
        return

    def fix_reference_energy(self,r,v,z):
        m = self.m
        F = self.pep
        E_kin = 0.5*torch.sum(m*v**2,dim=(1,2))
        with torch.no_grad():
            r_vec = r.reshape(-1, r.shape[-1])
            z_vec = z.reshape(-1, z.shape[-1])
            batch = torch.arange(r.shape[0]).repeat_interleave(r.shape[1]).to(device=r.device)
            E_pot = F(r_vec,z_vec,batch)
        E = E_kin + E_pot
        self.E0 = E.mean()
        return


    def Energy(self,r,v, batch, z,save_E_grad=False):
        m = self.m
        F = self.pep
        e = torch.ones((3,1),dtype=v.dtype,device=v.device)
        E_kin = 0.5*torch.sum(m*v**2,dim=(1,2)) #v should be in [Angstrom / fs]
        r_vec = r.reshape(-1, r.shape[-1])
        z_vec = z.reshape(-1, z.shape[-1])
        E_pot = F(r_vec,z_vec,batch).squeeze()
        E = E_pot + E_kin
        if save_E_grad:
            E_grad = grad(torch.sum(E_pot), r, create_graph=True)[0].requires_grad_(True) # This goes well since the kinetic energy does not depend on r and since r is not connected across different batches
            self.E_grad = E_grad
        return E

    def constraint(self,r,v,batch,z,save_E_grad):
        m = self.m
        E1 = self.Energy(r,v,batch, z,save_E_grad=save_E_grad)
        E = E1 - self.E0
        # P = v.transpose(1,2) @ m
        P = torch.sum(v * m,dim=1)
        c = torch.cat([E[:,None],P],dim=1)
        return c

    def dConstraintT(self,r,v,c):
        m = self.m
        E = c[:,0][:,None,None]
        P = c[:,1:][:,:,None]
        p = v * m
        e3 = torch.ones((3,1),dtype=v.dtype,device=v.device)
        en = torch.ones((v.shape[1],1),dtype=v.dtype,device=v.device)
        jr = self.E_grad * E
        jv = p * E + m @ e3.T * (en @ P.transpose(1,2))
        j = torch.cat([jr,jv],dim=-1)
        return j

    def forward(self, data, n=100, debug=True, converged=1e-3):
        y = data['y']
        batch = data['batch']
        z = data['z']
        nb = batch.max() + 1

        for j in range(n):
            x = self.project(y)
            r = x[:, 0:3].view(nb,-1,3)*self.rescale_r
            v = x[:, 3:].view(nb,-1,3)*self.rescale_v
            c = self.constraint(r,v,batch,z,save_E_grad=True)
            lam_x = self.dConstraintT(r,v,c)
            cnorm = c.norm(dim=1).mean()
            lam_x[...,0:3] = lam_x[...,0:3] / self.rescale_r
            lam_x[...,3:6] = lam_x[...,3:6] / self.rescale_v
            lam_y = self.uplift(lam_x.view(-1,6))
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_y.norm()
                    # alpha = 1.0
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:3].view(nb, -1, 3)*self.rescale_r
                    v = x[:, 3:].view(nb, -1, 3)*self.rescale_v
                    ctry = self.constraint(r,v,batch,z,save_E_grad=False)
                    ctry_norm = ctry.norm(dim=1).mean()
                    if ctry_norm < cnorm:
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        print("line search not working")
                        break
                if lsiter == 0 and ctry_norm > converged:
                    alpha = alpha * 1.5
            y = y - alpha * lam_y
            if debug:
                print(f"{self._get_name()} constraints {j} c: {cnorm.detach().cpu():2.8f} -> {ctry_norm.detach().cpu():2.8f}   ")
            if ctry_norm < converged:
                break
        return  {'y':y,'batch':batch, 'z':z}

