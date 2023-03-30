import math
import multibodypendulum as mbp
import torch
import numpy as np
import time
def create_and_save_multibodypendulum_data(nsamples,npenduls,dt,file):
    g = 9.82
    model = mbp.MultiBodyPendulum(npenduls, dt,g=g)
    theta0 = 0.5*math.pi*torch.ones(npenduls)
    dtheta0 = 0.0*torch.ones(npenduls)

    t1 = time.perf_counter()
    times, thetas, dthetas = model.simulate(nsamples,theta0,dtheta0)
    t2 = time.perf_counter()
    print(f"Simulated {nsamples} samples for a {npenduls}-body pendulum in {t2-t1:2.2f}s")
    model.plot_energy()
    ed = model.energy_drift()
    print(f"Average energy drift = {ed}")
    assert ed < 0.1

    x,y = model.xy
    vx,vy = model.vxy
    np.savez(file,theta=thetas,dtheta=dthetas,dt=dt,g=g,npenduls=npenduls,nsamples=nsamples,energy_drift=ed)
    return


if __name__ == '__main__':
    nsamples = 100000
    npenduls = 5
    dt = 0.001
    file = '../data/multibodypendulum/multibodypendulum.npz'
    create_and_save_multibodypendulum_data(nsamples,npenduls,dt,file)
# models.animate_pendulum()
