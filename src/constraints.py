import torch
import torch.nn as nn


class BindingConstraintsAlphaBetaN(nn.Module):
    def __init__(self,d_ab, d_an):
        super(BindingConstraintsAlphaBetaN, self).__init__()
        self.d_ab = d_ab
        self.d_an = d_an
        self.project = None
        self.uplift = None
        return

    def set_projectuplift(self,project,uplift):
        self.project = project
        self.uplift = uplift

    def constraint(self,xa,xb,xn):
        e = torch.ones(1,3,device=xa.device)
        dxa = self.diff(xa)
        c = e @ (dxa**2) - self.d**2
        return c

    def dConstraintT(self,c,x):
        dx = self.diff(x)
        e = torch.ones(3, 1, device=x.device)
        C = (e @ c) * dx
        C = self.diffT(C)
        return 2 * C

    def forward(self, y, batch, n=10, debug=False, converged=1e-4):
        for j in range(n):
            x = self.project(y)
            nvec = x.shape[-1]//2
            lam_x = torch.empty_like(x)
            nb = batch.max() + 1
            r = x[:, 0:nvec].view(nb, -1, nvec)
            v = x[:, nvec:].view(nb, -1, nvec)
            cnorm = 0
            for fragi in self.fragid_unique:
                idx = self.fragid == fragi
                ri = r[:,idx,:]
                ci = self.constraint(ri)
                lam_xi = self.dConstraintT(ri, ci)
                lam_x[idx] = lam_xi
                cnormi = torch.norm(ci)
                cnorm = cnorm + cnormi
            lam_y = self.uplift(lam_x.view(-1, 6))
            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam_y.norm()
                lsiter = 0
                while True:
                    ytry = y - alpha * lam_y
                    x = self.project(ytry)
                    r = x[:, 0:3].view(nb, -1, 3)
                    v = x[:, 3:].view(nb, -1, 3)
                    ctry = 0
                    for fragi in self.fragid_unique:
                        idx = self.fragid == fragi
                        ri = r[:,idx,:]
                        ctry = ctry + self.constraint(ri)
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
                print(f"{j} c: {cnorm.detach().cpu():2.4f} -> {ctry.detach().cpu():2.4f}   ")
            if ctry_norm < converged:
                break
        return y


class BindingConstraintsNN(nn.Module):
    def __init__(self,distance,fragmentid):
        super(BindingConstraintsNN, self).__init__()
        self.d = distance
        self.fragid = fragmentid
        self.fragid_unique = torch.unique(fragmentid)
        self.project = None
        self.uplift = None
        return

    def set_projectuplift(self,project,uplift):
        self.project = project
        self.uplift = uplift

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
        c = (dx**2)@e - self.d**2
        return c

    def dConstraintT(self,c,x):
        dx = self.diff(x)
        e = torch.ones(1, 3, device=x.device)
        C = (c @ e) * dx
        C = self.diffT(C)
        return 2 * C

    def forward(self, y, batch, n=10, debug=False, converged=1e-4):
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
                    alpha = 1.0 / lam_y.norm()
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
                print(f"NN constraints {j} c: {cnorm.detach().cpu():2.4f} -> {ctry_norm.detach().cpu():2.4f}   ")
            if ctry_norm < converged:
                break
        return y

class BindingConstraintsAB(nn.Module):
    def __init__(self,d_ab,d_an,fragmentid):
        super(BindingConstraintsAB, self).__init__()
        self.d_ab = d_ab
        self.d_an = d_an
        self.fragid = fragmentid
        self.fragid_unique = torch.unique(fragmentid)
        self.project = None
        self.uplift = None
        return

    def set_projectuplift(self,project,uplift):
        self.project = project
        self.uplift = uplift

    def diff(self,x,x0):
        return x - x0

    def diffT(self,dx,x0):
        return dx+x0

    def constraint(self,x,x0,d):
        e = torch.ones((3,1),device=x.device)
        dx = self.diff(x,x0)
        c = torch.sqrt(d**2/(dx**2@e)) - 1
        return c

    def dConstraintT(self,c,x,x0):
        dx = self.diff(x,x0)
        lam = - dx * c
        return lam

    def forward(self, y, batch, n=10, debug=False, converged=1e-4):
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
                ri0 = ri[:, :,:3]
                rib = ri[:, :,3:6]
                rin = ri[:, :,6:9]
                cib = self.constraint(rib,ri0,self.d_ab[idx][None,:,None])
                cin = self.constraint(rin,ri0,self.d_an[idx][None,:,None])
                lam_xib = self.dConstraintT(cib, rib,ri0)
                lam_xin = self.dConstraintT(cin, rin,ri0)
                lam_x[idx,3:6] = lam_xib
                lam_x[idx,6:9] = lam_xin
                cnormib = torch.sum(cib**2)
                cnormin = torch.sum(cin**2)
                cnorm = cnorm + cnormib + cnormin
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
                        ri0 = ri[:, :, :3]
                        rib = ri[:, :, 3:6]
                        rin = ri[:, :, 6:9]
                        ctryb = self.constraint(rib, ri0,self.d_ab[idx][None,:,None])
                        ctryn = self.constraint(rin, ri0,self.d_an[idx][None,:,None])
                        ctry_norm = ctry_norm + torch.sum(ctryb**2) + torch.sum(ctryn**2)
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
                print(f"ABN constraints {j} c: {cnorm.detach().cpu():2.4f} -> {ctry_norm.detach().cpu():2.4f}   ")
            if ctry_norm < converged:
                break
        return y


class MomentumConstraints(nn.Module):
    def __init__(self,m,project,uplift):
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

    def forward(self, y, batch, n=10, debug=False, converged=1e-4):
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
                print(f"{j} c: {c.detach().cpu().norm():2.4f} -> {ctry.detach().cpu().norm():2.4f}   ")
            if ctry_norm < converged:
                break
        return y
