import torch
import torch.nn as nn
from torch.autograd import grad

from preprocess.train_force_and_energy_predictor import generate_FE_network


def load_constraints(ctype,PU,masses=None,R=None,V=None,z=None,rscale=1,vscale=1,energy_predictor=None):
    if ctype == 'chain':
        constraints = torch.nn.Sequential(PointChain(PU.project,PU.uplift,3.8, fragmentid=fragids))
    elif ctype == 'triangle':
        constraints = torch.nn.Sequential(PointToPoint(PU.project,PU.uplift,r=dist_abz.to(device)),PointToSphereSphereIntersection(PU.project,PU.uplift,r1=dist_anz.to(device),r2=dist_bnz.to(device)))
    elif ctype == 'chaintriangle':
        constraints = torch.nn.Sequential(PointChain(PU.project,PU.uplift,3.8, fragmentid=fragids),PointToPoint(PU.project,PU.uplift,r=dist_abz.to(device)),PointToSphereSphereIntersection(PU.project,PU.uplift,r1=dist_anz.to(device),r2=dist_bnz.to(device)))
        # constraints2 = BindingConstraintsAB(d_ab=dist_abz.to(device), d_an=dist_anz.to(device), fragmentid=fragids)
    elif ctype == 'P':
        constraints = torch.nn.Sequential(MomentumConstraints(PU.project, PU.uplift, masses))
    elif ctype == 'EP':
        force_predictor = generate_FE_network(natoms=z.shape[1])
        force_predictor.load_state_dict(torch.load(energy_predictor, map_location=torch.device('cpu')))
        force_predictor.eval()
        constraints = torch.nn.Sequential(EnergyMomentumConstraints(PU.project, PU.uplift,force_predictor, masses,rescale_r=rscale,rescale_v=vscale))
        constraints[0].fix_reference_energy(R,V,z)
    elif ctype == '':
        constraints = None
    else:
        raise NotImplementedError("The constraint chosen has not been implemented.")
    return constraints

class PointToPoint(nn.Module):
    def __init__(self,project,uplift,r):
        super(PointToPoint, self).__init__()
        self.r = r
        self.project = project
        self.uplift = uplift
        return

    def constraint(self,p1,p2):
        r = self.r
        d = p2 - p1
        lam = d * (1 - (r[None,:,None] / d.norm(dim=-1).unsqueeze(-1)))
        return lam

    def constrain_high_dim(self,data,debug=False):
        batch = data['batch']
        nb = batch.max() + 1
        y = data['y']
        # for j in range(n):
        x = self.project(y)
        ndim = x.shape[-1] // 2
        nvec = ndim // 3
        lam_x = torch.zeros_like(x)
        nb = batch.max() + 1
        r = x[:, 0:ndim].view(nb, -1, ndim)
        v = x[:, ndim:].view(nb, -1, ndim)
        lam_p2 = self.constraint(r[:, :, 0:3], r[:, :, 3:6])
        lam_x[:, 3:6] = lam_p2
        lam_y = self.uplift(lam_x.view(-1, 2 * ndim))
        alpha = 1
        y = y - alpha * lam_y
        if debug:
            x = self.project(y)
            r = x[:, 0:ndim].view(nb, -1, ndim)
            lam_p2_after = self.constraint(r[:, :, 0:3], r[:, :, 3:6])
            cnorm = torch.mean(torch.sum(lam_p2 ** 2, dim=-1))
            cnorm_after = torch.mean(torch.sum(lam_p2_after ** 2, dim=-1))
            print(f"{self._get_name()} constraint c: {cnorm:2.4f} -> {cnorm_after:2.4f}")
        return {'y': y, 'batch': batch}

    def constrain_low_dim(self,data):
        batch = data['batch']
        nb = batch.max() + 1
        x = data['x']
        ndim = x.shape[-1] // 2
        r = x[:, 0:ndim].view(nb, -1, ndim)
        v = x[:, ndim:].view(nb, -1, ndim)
        lam_p2 = self.constraint(r[:, :, 0:3], r[:, :, 3:6])
        x[:,3:6] = x[:,3:6] + lam_p2
        return {'x': x, 'batch': batch}


    def forward(self, data, n=10, debug=True, converged=1e-4):
        if 'x' in data.files:
            data = self.constrain_low_dim(data)
        else:
            data = self.constrain_high_dim(data)
        return data

class PointToSphereSphereIntersection(nn.Module):
    def __init__(self,project,uplift,r1,r2):
        super(PointToSphereSphereIntersection, self).__init__()
        self.r1 = r1
        self.r2 = r2
        self.project = project
        self.uplift = uplift
        return

    def constraint(self,p1,p2,p3):
        r1 = self.r1
        r2 = self.r2
        d = p2 - p1
        dn = d.norm(dim=-1)
        a = 1 / (2 * dn) * torch.sqrt(4 * dn ** 2 * r2 ** 2 - (dn ** 2 - r1 ** 2 + r2 ** 2))

        cn = (dn ** 2 - r2 ** 2 + r1 ** 2) / (2 * dn)
        c = cn[:,:,None] / dn[:,:,None] * d + p1
        n = d / dn.unsqueeze(-1)

        q = p3 - c - (torch.sum(n*(p3 - c),dim=-1,keepdim=True) * n)
        K = c + a.unsqueeze(-1) * q / q.norm(dim=-1).unsqueeze(-1)

        lam_p3 = K - p3
        return lam_p3

    def constrain_low_dim(self,data):
        x = data['y']
        batch = data['batch']
        ndim = x.shape[-1]//2
        nvec = ndim // 3
        lam_x = torch.zeros_like(x)
        nb = batch.max() + 1
        r = x[:, 0:ndim].view(nb, -1, ndim)
        v = x[:, ndim:].view(nb, -1, ndim)

        lam_p3 = self.constraint(r[:,:,0:3],r[:,:,3:6],r[:,:,6:9])
        x[:,6:9] = x[:,6:9] + lam_p3
        return {'x':x,'batch':batch}

    def constrain_high_dim(self,data,debug=False,n=10,converged=1E-4):
        y = data['y']
        batch = data['batch']
        for j in range(n):
            x = self.project(y)
            ndim = x.shape[-1]//2
            nvec = ndim // 3
            lam_x = torch.zeros_like(x)
            nb = batch.max() + 1
            r = x[:, 0:ndim].view(nb, -1, ndim)
            v = x[:, ndim:].view(nb, -1, ndim)

            lam_p3 = self.constraint(r[:,:,0:3],r[:,:,3:6],r[:,:,6:9])
            lam_x[:,6:9] = lam_p3

            lam_y = self.uplift(lam_x.view(-1, 2*ndim))
            cnorm = torch.sum(lam_p3**2)
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 #/ lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y + alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:ndim].view(nb, -1, ndim)
                    v = x[:, ndim:].view(nb, -1, ndim)
                    lam_p3_try = self.constraint(r[:, :,0:3], r[:, :,3:6], r[:, :,6:9])
                    ctry_norm = torch.sum(lam_p3_try**2)
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

    def forward(self, data, n=10, debug=True, converged=1e-4):
        if 'x' in data.files:
            data = self.constrain_low_dim(data)
        else:
            data = self.constrain_high_dim(data)
        return data



class PointChain(nn.Module):
    def __init__(self,project,uplift,distance,fragmentid):
        super(PointChain, self).__init__()
        self.project = project
        self.uplift = uplift
        self.d = distance
        self.fragid = fragmentid
        self.fragid_unique = torch.unique(fragmentid)
        return

    def diff(self,x):
        return x[:,1:] - x[:,:-1]

    def diffT(self,dx):
        x = dx[:,:-1] - dx[:,1:]
        x0 = -dx[:,:1]
        x1 = dx[:,-1:]
        X = torch.cat([x0,x,x1],dim=1)
        return X

    def constraint(self,x):
        e = torch.ones((3,1),device=x.device)
        dx = self.diff(x)
        c = self.d/torch.sqrt((dx**2)@e) - 1
        return c

    def dConstraintT(self,c,x):
        dx = self.diff(x)
        drn2 = torch.sum(dx**2,dim=2,keepdim=True)
        lam0 = self.d*drn2[:,:1]**(-3/2)*c[:,0]*dx[:,:1]
        lam1 = -self.d*drn2[:,-1:]**(-3/2)*c[:,-1]*dx[:,-1:]
        lam01 = -self.d*drn2[:,:-1]**(-3/2)*c[:,:-1]*dx[:,:-1] + self.d*drn2[:,1:]**(-3/2)*c[:,1:]*dx[:,1:]
        lam = torch.cat([lam0,lam01,lam1],dim=1)
        return lam

    def forward(self, data, n=10, debug=True, converged=1e-4):
        y = data['y']
        batch = data['batch']
        for j in range(n):
            x = self.project(y)
            ndim = x.shape[-1]//2
            nvec = ndim // 3
            lam_x = torch.zeros_like(x)
            nb = batch.max() + 1
            r = x[:, 0:ndim].view(nb, -1, ndim)
            v = x[:, ndim:].view(nb, -1, ndim)
            cnorm = 0
            for fragi in self.fragid_unique:
                idx = self.fragid == fragi
                ri = r[:,idx,:]
                ria = ri[:, :,:3]
                ci = self.constraint(ria)
                lam_xia = self.dConstraintT(ci, ria)
                lam_xi = lam_xia.repeat(1,1,nvec)
                lam_x[idx,:ndim] = lam_xi
                cnormi = torch.sum(ci**2)
                cnorm = cnorm + cnormi
            lam_y = self.uplift(lam_x.view(-1, 2*ndim))
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 #/ lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:ndim].view(nb, -1, ndim)
                    v = x[:, ndim:].view(nb, -1, ndim)
                    ctry_norm = 0
                    for fragi in self.fragid_unique:
                        idx = self.fragid == fragi
                        ri = r[:,idx,:]
                        ria = ri[:, :, :3]
                        ctry = self.constraint(ria)
                        ctry_norm = ctry_norm + torch.sum(ctry**2)
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

#
# class EnergyMomentumConstraints(nn.Module):
#     def __init__(self,project,uplift,m,potential_energy_predictor):
#         super(EnergyMomentumConstraints, self).__init__()
#         self.register_buffer("m", m[:,None])
#         self.project = project
#         self.uplift = uplift
#         self.pep = potential_energy_predictor
#         self.pep.eval()
#         return
#
#     def fix_reference_energy(self,r,v):
#         m = self.m
#         F = self.pep
#         E_kin = 0.5*m*torch.sum(v**2,dim=-1)
#         with torch.no_grad:
#             E_pot = F(r)
#         E = E_kin + E_pot
#         self.E0 = E.mean()
#         return
#
#     def Energy(self,m,r,v,return_gradient=False):
#         F = self.pep
#         m = self.m
#         E_kin = 0.5*m*torch.sum(v**2,dim=-1)
#         E_pot = F(r)
#         E = E_pot + E_kin
#         if return_gradient:
#             E_grad = grad(E, r, create_graph=False)[0]
#             return E, E_grad
#         else:
#             return E
#
#     def constraint(self,m,r,v,return_gradient):
#         E1, E_grad = self.Energy(m,r,v,return_gradient=return_gradient)
#         E = E1 - self.E0
#         p = m @ v
#         # c = torch.cat([E,p],dim=-1)
#         return E, p, E_grad
#
#     def dConstraintT(self,m,E,p,E_grad):
#         J1 = E_grad * E
#         J2 = p * E + m * p
#         J = torch.cat([J1,J2],dim=-1)
#         return J
#
#     # def constraint(self,v):
#     #     m = self.m
#     #     P = v.transpose(1,2) @ m
#     #     return P
#     #
#     def dConstraintT(self,v,c):
#         m = self.m
#         e3 = torch.ones((3,1),dtype=v.dtype,device=v.device)
#         en = torch.ones((m.shape[0],1),dtype=v.dtype,device=v.device)
#         jv = m @ e3.T * (en @ c.transpose(1,2))
#         jr = torch.zeros_like(jv)
#         j = torch.cat([jr,jv],dim=-1)
#         return j
#
#     def forward(self, data, n=10, debug=False, converged=1e-4):
#         y = data['y']
#         batch = data['batch']
#
#         for j in range(n):
#             x = self.project(y)
#             nb = batch.max()+1
#             r = x[:, 0:3].view(nb,-1,3)
#             v = x[:, 3:].view(nb,-1,3)
#             c = self.constraint(v)
#             lam_x = self.dConstraintT(v,c)
#             cnorm = torch.norm(c)
#             lam_y = self.uplift(lam_x.view(-1,6))
#             with torch.no_grad():
#                 if j == 0:
#                     alpha = 1.0 / lam_y.norm()
#                 lsiter = 0
#                 while True:
#                     ytry = y - alpha * lam_y
#                     x = self.project(ytry)
#                     r = x[:, 0:3].view(nb, -1, 3)
#                     v = x[:, 3:].view(nb, -1, 3)
#                     ctry = self.constraint(v)
#                     ctry_norm = torch.norm(ctry)
#                     if ctry_norm < cnorm:
#                         break
#                     alpha = alpha / 2
#                     lsiter = lsiter + 1
#                     if lsiter > 10:
#                         break
#                 if lsiter == 0 and ctry_norm > converged:
#                     alpha = alpha * 1.5
#             y = y - alpha * lam_y
#             if debug:
#                 print(f"{j} c: {c.detach().cpu().norm():2.4f} -> {ctry.detach().cpu().norm():2.4f}   ")
#             if ctry_norm < converged:
#                 break
#         return {'y':y,'batch':batch}
#
# #
#
# class EnergyMomentumConstraints(nn.Module):
#     def __init__(self,potential_energy_predictor,m):
#         super(EnergyMomentumConstraints, self).__init__()
#         self.F = potential_energy_predictor
#         self.E0 = None
#         return
#
#     def Energy(self,m,r,v,save_E0=False,return_gradient=False):
#         F = self.F
#         E_kin = 0.5*m*torch.sum(v**2,dim=-1)
#         E_pot = F(r)
#         E = E_pot + E_kin
#         if save_E0:
#             self.E0 = E
#         if return_gradient:
#             E_grad = grad(E, r, create_graph=False)[0]
#             return E, E_grad
#         else:
#             return E
#
#     def constraint(self,m,r,v,return_gradient):
#         E1, E_grad = self.Energy(m,r,v,return_gradient=return_gradient)
#         E = E1 - self.E0
#         p = m @ v
#         # c = torch.cat([E,p],dim=-1)
#         return E, p, E_grad
#
#     def dConstraintT(self,m,E,p,E_grad):
#         J1 = E_grad * E
#         J2 = p * E + m * p
#         J = torch.cat([J1,J2],dim=-1)
#         return J
#
#     def forward(self,m,r,v):
#         E,p, E_grad = self.constraint(m,r,v,return_gradient=True)
#         c = torch.cat([E, p], dim=-1)
#         J = self.dConstraintT(m,E,p,E_grad)
#         return c, J