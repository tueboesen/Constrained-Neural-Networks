import math
import torch
import numpy as np
from src.vizualization import plot_pendulum_snapshot, plot_pendulum_snapshots, plot_pendulum_snapshot_custom

torch.set_default_dtype(torch.float64)
import time


class NPendulum:
    """
    A multibody pendulum class, which enables the simulation of a pendulum with massless rigid strings between n point masses.

    Note that this code only supports pendulums where all pendulums have a length of 1.
    """
    def __init__(self, npendulums: int, dt: float, g: float = 9.82):
        """
        Initiates an n-pendulum class for simulating multi-body pendulums.
        See https://travisdoesmath.github.io/pendulum-explainer/ or https://github.com/tueboesen/n-pendulum for the equations,
        :param npendulums: number of pendulums
        :param dt: time stepping size
        """
        self.n = npendulums
        self.g = g
        self.dt = dt

        self.A = torch.zeros((npendulums, npendulums)) #Memory allocations
        self.b = torch.zeros((npendulums))
        self.x = torch.zeros((npendulums))


        C = torch.zeros(npendulums, npendulums) # A lookup matrix that we use for fast calculation of static information rather than actually compute this in every step
        for i in range(npendulums):
            for j in range(npendulums):
                C[i,j] = self.c(i,j)
        self.C = C

    def c(self,i: int,j: int):
        return self.n - max(i,j)

    def step(self, theta, dtheta):
        """
        Performs a single step for a multibody pendulum.

        More specifically it sets up and solves the n-pendulum linear system of equations in angular coordinates where the constraints are naturally obeyed.

        Note this function should not be used. Use npendulum_fast instead since it is an optimized version of this. This function is still here for comparison and to easily understand the calculations.

        :param theta: angular coordinate
        :param dtheta: angular velocity
        :return:
        """
        n = self.n
        A = self.A
        b = self.b
        g = self.g

        for i in range(n):
            for j in range(n):
                A[i, j] = self.c(i, j) * torch.cos(theta[i]-theta[j])

        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp = tmp - self.c(i,j) * dtheta[j]*dtheta[j] * torch.sin(theta[i] - theta[j])
            b[i] = tmp - g * (n - i + 1) *torch.sin(theta[i])

        ddtheta = torch.linalg.solve(A,b)
        return dtheta,ddtheta

    def step_fast(self, theta, dtheta):
        """
        Faster version of step
        """
        n = self.n
        g = self.g

        A = self.C * torch.cos(theta[:,None] - theta[None,:])

        B = - self.C * (dtheta[None,:]*dtheta[None,:]) * torch.sin(theta[:,None] - theta[None,:])
        b = torch.sum(B,dim=1)
        b = b - g * torch.arange(n,0,-1) * torch.sin(theta)
        ddtheta = torch.linalg.solve(A,b)
        return dtheta,ddtheta

    def rk4(self,dt,theta,dtheta):
        """
        Runge kutta 4 integration
        :param dt:
        :param theta:
        :param dtheta:
        :return:
        """
        k1 = self.step_fast(theta, dtheta)
        k2 = self.step_fast(theta + dt / 2 * k1[0], dtheta + dt / 2 * k1[1])
        k3 = self.step_fast(theta + dt / 2 * k2[0], dtheta + dt / 2 * k2[1])
        k4 = self.step_fast(theta + dt * k3[0], dtheta + dt * k3[1])

        theta  = theta + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        dtheta = dtheta + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        return theta,dtheta

    def simulate(self,nsteps: int, theta_start: torch.FloatTensor, dtheta_start: torch.FloatTensor):
        """
        Simulates an n-pendulum.
        :param nsteps:
        :param theta:
        :param dtheta:
        :return:
        """
        dt = self.dt
        n = self.n
        thetas = torch.zeros(n,nsteps+1)
        dthetas = torch.zeros(n,nsteps+1)
        thetas[:, 0] = theta_start
        dthetas[:, 0] = dtheta_start
        t = torch.linspace(0,nsteps * dt, nsteps + 1)
        theta = theta_start
        dtheta = dtheta_start

        for i in range(nsteps):
            theta, dtheta = self.rk4(dt,theta,dtheta)
            thetas[:,i+1] = theta
            dthetas[:,i+1] = dtheta
        return t, thetas, dthetas

def get_coordinates_from_angle(thetas: torch.FloatTensor, dthetas: torch.FloatTensor):
    """
    Converts angles to cartesian coordinates and velocities.
    Expects thetas and dthetas to have the shape [npendulums,...]
    """
    assert thetas.shape == dthetas.shape
    n = thetas.shape[0]
    x = torch.empty_like(thetas)
    y = torch.empty_like(thetas)
    vx = torch.empty_like(thetas)
    vy = torch.empty_like(thetas)

    x[0] = torch.sin(thetas[0])
    y[0] = - torch.cos(thetas[0])
    vx[0] = dthetas[0] * torch.cos(thetas[0])
    vy[0] = dthetas[0] * torch.sin(thetas[0])
    for i in range(1,n):
        x[i] = x[i-1] + torch.sin(thetas[i])
        y[i] = y[i-1] - torch.cos(thetas[i])
        vx[i] = vx[i-1] + dthetas[i] * torch.cos(thetas[i])
        vy[i] = vy[i-1] + dthetas[i] * torch.sin(thetas[i])
    return x, y, vx, vy

def get_angles_from_coordinates(x,y,vx,vy):
    """
    Converts cartesian coordinates and velocities to angles.
    Expects the input to have shape [npendulums,...]
    """
    assert x.shape == y.shape == vx.shape == vy.shape
    thetas = torch.empty_like(x)
    dthetas = torch.empty_like(x)
    thetas[0] = torch.atan(y[0]/x[0])
    dthetas[0] = torch.atan(vy[0]/vx[0])
    for i in range(1,x.shape[0]):
        thetas[i] = torch.atan((y[i]-y[i-1])/(x[i]-x[i-1]))
        dthetas[i] = torch.atan((vy[i]-vy[i-1])/(vx[i]-vx[i-1]))
    return thetas, dthetas



if __name__ == '__main__':
    n = 5
    dt = 0.001
    Npend = NPendulum(n,dt)
    g = Npend.g
    theta0 = 0.5*math.pi*torch.ones(n)
    dtheta0 = 0.0*torch.ones(n)
    nsteps = 10000

    t0 = time.time()
    times, thetas, dthetas = Npend.simulate(nsteps,theta0,dtheta0)
    t1 = time.time()
    print(f"simulated {nsteps} steps for a {n}-pendulum in {t1-t0:2.2f}s")

    x,y,vx,vy = get_coordinates_from_angle(thetas,dthetas)
    v2 = vx**2 + vy**2
    K = 0.5*torch.sum(v2[1:],dim=0)
    V = g * torch.sum(y[1:],dim=0)
    E = K + V
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plt.figure(1)
    plt.plot(E,label='Energy')
    plt.plot(K,label='Kinetic energy')
    plt.plot(V,label='Potential energy')
    plt.legend()
    plt.show()
    plt.pause(1)

    plt.figure(2)
    plt.plot(E,label='Energy')
    plt.legend()
    plt.show()
    plt.pause(1)

    # x = x[:,100:]
    # y = y[:,100:]
    # vx = vx[:,100:]
    # vy = vy[:,100:]
    # filename = f"{output_folder}/viz/{epoch}_{j}_{'train' if train == True else 'val'}_.png"
    # plot_pendulum_snapshot(Rin_xy[j], Rout_xy[j], Vin_xy[j], Vout_xy[j], Rpred_xy[j], Vpred_xy[j], file=filename)
    ii = [150,250,400,500,900,5100,5600,5800,1230,3245]
    # k = np.asarray([20,50,100])
    k = np.asarray([20,50])
    for i in ii:
        Rin = torch.cat((x.T[i,1:,None],y.T[i,1:,None]),dim=-1)
        Rout = torch.cat((x.T[i+k,1:,None],y.T[i+k,1:,None]),dim=-1)
        Vin = torch.cat((vx.T[i,1:,None],vy.T[i,1:,None]),dim=-1)
        Vout = torch.cat((vx.T[i+k,1:,None],vy.T[i+k,1:,None]),dim=-1)
        file = f'/home/tue/npendulum/npendulum_{i}_{k}.png'
        # plot_pendulum_snapshots(Rin, Rout, Vin, Vout, Rpred=None, Vpred=None, file=file)
        fighandler = plot_pendulum_snapshot_custom(Rin,Vin,color='red')
        fighandler = plot_pendulum_snapshot_custom(Rout[0],Vout[0],fighandler=fighandler,color='green')
        fighandler = plot_pendulum_snapshot_custom(Rout[1],Vout[1],fighandler=fighandler,file=file,color='blue')
        # plot_pendulum_snapshot_custom(Rin,Vin,fignum=1)


    # animate_pendulum(x.numpy(), y.numpy(),vx.numpy(),vy.numpy())

    # file = f'/home/tue/{n}_pendulum_{nsteps}.gif'
    # animate_pendulum(x.numpy(), y.numpy(),save=file)

