import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim

def double_pendulum_system(x,L1=3.0,L2=2.0,m1=2.0,m2=0.5,g=9.8):
# a system of differential equations defining a double pendulum
# from http://www.myphysicslab.com/dbl_pendulum.html
    theta1 = x[0]
    theta2 = x[1]
    omega1 = x[2]
    omega2 = x[3]
    dtheta1 = omega1
    dtheta2 = omega2
    domega1 = (-g*(2*m1+m2)*torch.sin(theta1)  -
               m2*g*torch.sin(theta1-2*theta2) -
               2*torch.sin(theta1-theta2)*m2*
               (omega2**2*L2+omega1**2*L1*torch.cos(theta1-theta2)))/(L1*(2*m1+m2-m2*torch.cos(2*theta1-2*theta2)))
    domega2 = (2*torch.sin(theta1-theta2)*(omega1**2*L1*(m1+m2) +
               g*(m1+m2)*torch.cos(theta1)+omega2**2*L2*m2*
               torch.cos(theta1-theta2)))/(L2*(2*m1+m2-m2*torch.cos(2*theta1-2*theta2)));
    f = torch.tensor([dtheta1, dtheta2,domega1, domega2])

    return f

def integrateSystem(x0,dt,N):

    x = x0
    X = torch.zeros(4,N+1)
    X[:,0] = x
    for i in range(N):
        k1 = double_pendulum_system(x)
        k2 = double_pendulum_system(x + dt/2*k1)
        k3 = double_pendulum_system(x + dt/2*k2)
        k4 = double_pendulum_system(x + dt*k3)
        x  = x + dt/6*(k1+2*k2+2*k3+k4)
        X[:, i+1] = x

    t = torch.linspace(0,N*dt,N+1)

    return X, t

def getCoordsFromAngles(x,L1=3.0,L2=2.0):

    theta1 = x[0]
    theta2 = x[1]
    xm1 = L1*torch.sin(theta1)
    ym1 = -L1*torch.cos(theta1)
    xm2 = xm1 + L2*torch.sin(theta2)
    ym2 = ym1 - L2*torch.cos(theta2)
    X = torch.cat((xm1.unsqueeze(0),ym1.unsqueeze(0),xm2.unsqueeze(0),ym2.unsqueeze(0)))
    return X

def getAnglesFromCoords(x,L1=3.0,L2=2.0):

    xm1 = x[0]
    xm2 = x[1]
    theta1 = torch.arcsin(xm1/L1)
    theta2 = torch.arcsin((xm2-xm1)/L2)

    return torch.tensor([[theta1],[theta2]])

theta0 = torch.tensor([3*np.pi/4,np.pi/2,0,0])
Theta, t = integrateSystem(theta0,0.01,10000)
X        = getCoordsFromAngles(Theta)

# Coord vs time plot

# plt.plot(t, X[0, :])
# plt.plot(t, X[1, :])
# plt.plot(t, X[2, :])
# plt.plot(t, X[3, :])
for j in range(5):  # range(X.shape[1]):
    t1 = torch.zeros(3);
    t1[1:] = X[:2, j]
    t2 = torch.zeros(3);
    t2[1:] = X[2:, j]

    plt.figure(1)
    plt.clf()

    plt.plot(t1, t2, 'o-')
    plt.show()
    plt.pause(0.1)


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))

def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


class hyperNet(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, Arch, h=0.1):
        super(hyperNet, self).__init__()
        Kopen, Kclose, W, Bias= self.init_weights(Arch)
        self.Kopen  = Kopen
        self.Kclose = Kclose
        self.W = W
        self.Bias = Bias
        self.h = h

    def init_weights(self,A):
        print('Initializing network  ')
        #Arch = [nstart, nopen, nhid, nclose, nlayers]
        nstart  = A[0]
        nopen   = A[1]
        nhid    = A[2]
        nclose  = A[3]
        nlayers = A[4]

        Kopen = torch.zeros(nopen, nstart)
        stdv  = 1e-3 * Kopen.shape[0]/Kopen.shape[1]
        Kopen.data.uniform_(-stdv, stdv)
        Kopen = nn.Parameter(Kopen)

        Kclose = torch.zeros(nclose, nopen)
        stdv = 1e-3 * Kclose.shape[0] / Kclose.shape[1]
        Kclose.data.uniform_(-stdv, stdv)
        Kclose = nn.Parameter(Kclose)

        W = torch.zeros(nlayers, nhid, nopen, 3)
        stdv = 1e-4
        W.data.uniform_(-stdv, stdv)
        W = nn.Parameter(W)

        Bias = torch.rand(nlayers,nopen,1)*1e-4
        Bias = nn.Parameter(Bias)

        return Kopen, Kclose, W, Bias

    def doubleSymLayer(self, Z, Wi, Bi):
        Ai = conv1(Z + Bi, Wi)
        Ai = F.instance_norm(Ai)
        Ai = F.instance_norm(Ai)

        # Layer T
        Ai = conv1T(Ai, Wi)

        return Ai

    def forward(self, Z, m=1.0):

        h = self.h
        l = self.W.shape[0]
        Kopen = self.Kopen
        Kclose = self.Kclose
        Z = Kopen@Z
        Zold = Z
        for i in range(l):

            Wi = self.W[i]
            Bi = self.Bias[i]
            # Layer
            Ai = self.doubleSymLayer(Z, Wi, Bi)
            Ztemp = Z
            Z = 2*Z - Zold - (h**2)*Ai
            Zold = Ztemp
            # for non hyperbolic use
            # Z = Z - h*Ai.squeeze(0)
        # closing layer back to desired shape
        Z    = Kclose@Z
        Zold = Kclose@Zold
        return Z, Zold


    def NNreg(self):

        dWdt = self.W[1:] - self.W[:-1]
        RW   = torch.sum(torch.abs(dWdt))/dWdt.numel()
        RKo  = torch.norm(self.Kopen)**2/2/self.Kopen.numel()
        RKc = torch.norm(self.Kclose)**2/2/self.Kclose.numel()
        return RW + RKo + RKc


theta0 = torch.tensor([3*np.pi/4,np.pi/2,0,0])
Theta, t = integrateSystem(theta0,0.01,10000)
X        = getCoordsFromAngles(Theta)

# Coord vs time plot
plt.figure(1)
plt.plot(t, X[0, :])
plt.plot(t, X[1, :])
plt.plot(t, X[2, :])
plt.plot(t, X[3, :])

# phase space plot
plt.figure(2)
plt.plot(X[2, :], X[3, :],'.b')
plt.plot(X[0, :], X[1, :],'.r')
plt.savefig('phasespace_pendulum.png')
plt.xlabel("x")
plt.ylabel("y")
print('done')
# Data for learning system
# Divide the data into I/O
def getData(X,idx=16,idxGap=100):

    n = X.shape[1]
    k = n-idx-idxGap
    X0 = torch.zeros(k,4,idx)
    XT = torch.zeros(k,4)
    for i in range(k):
        X0[i,:,:] = X[:,i:i+idx]
        XT[i,:] = X[:,i+idx+idxGap]

    return X0, XT

idx    = 16
idxGap = 100
X0, XT = getData(X,idx,idxGap)

# Separate to training/Validation
nd = 9500
X0val = X0[nd:,:,:]
XTval = XT[nd:,:]
X0    = X0[:nd,:,:]
XT    = XT[:nd,:]

# Now build a network
nstart  = 4
nopen   = 32
nhid    = 64
nclose  = 4
nlayers = 18
h       = 1/nlayers
Arch = [nstart, nopen, nhid, nclose, nlayers]
model = hyperNet(Arch,h)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ',total_params)

lrO = 1e-3
lrC = 1e-3
lrN = 1e-3
lrB = 1e-3
lrv = 1e-3

epochs = 500
batch_size = 100
n_batches = X0.shape[0]//batch_size
v = nn.Parameter(1e-4*torch.randn(idx))
optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.W, 'lr': lrN},
                        {'params': model.Bias, 'lr': lrB},
                        {'params': v, 'lr': lrv}])

print('         Misfit       Reg          gradv        gradW       gradKo        gradKc       gradB')
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    for i in range(n_batches):
        jj = i*batch_size
        X0i = X0[jj:jj+batch_size,:,:]
        XTi = XT[jj:jj+batch_size,:]
        optimizer.zero_grad()
        # From Coords to Seq
        Yi, _ = model(X0i)
        XTci  = Yi@v
        misfit = F.mse_loss(XTci, XTi)/F.mse_loss(XTi*0, XTi)
        R = model.NNreg()
        loss = misfit +  R

        loss.backward(retain_graph=True)

        aloss += loss.detach()
        amis += misfit.detach().item()

        optimizer.step()
        nprnt = 5
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" %
                  (j, i, amis, R.item(),model.W.grad.norm().item(),
                   model.W.grad.norm().item(), model.Kopen.grad.norm().item(),
                   model.Kclose.grad.norm().item(),
                   model.Bias.grad.norm().item()))
            amis = 0.0
            aloss = 0.0

    # Validation on 0-th data
    with torch.no_grad():
        Yval, _ = model(X0val)
        XTci = Yval @ v
        misfit = F.mse_loss(XTci, XTval)/F.mse_loss(XTval*0, XTval)
        print("%2d     %10.3E" % (j, misfit))
        print('===============================================')
