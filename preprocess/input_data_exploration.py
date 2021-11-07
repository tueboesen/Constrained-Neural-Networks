"""
This preprocessing file plots the average change in distance and velocity over a range of nskips. This is helpful in order to get an idea about whether a prediction difference is statistically significant.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.utils import convert_snapshots_to_future_state_dataset, Distogram
import torch

folder = '/home/tue/data/MD/ethanol_heating/'
name = 'ethanol.npz'
filename = folder + name

data = np.load(filename)

R = torch.from_numpy(data['R'])
V = torch.from_numpy(data['V'])
T = torch.from_numpy(data['temp'])
nskips = torch.linspace(0,R.shape[0]-1,500).round().to(dtype=torch.int64)


dRhist = []
dVhist = []
MAErhist = []
for i in range(nskips.shape[0]):
    nskip = nskips[i]
    Rin,Rout = convert_snapshots_to_future_state_dataset(nskip,R)
    Vin,Vout = convert_snapshots_to_future_state_dataset(nskip,V)


    DRin = Distogram(Rin)
    DRout = Distogram(Rout)

    DVin = Distogram(Vin)
    DVout = Distogram(Vout)

    dR = torch.mean(torch.sum(torch.abs(DRin - DRout), dim=(1,2)))
    dV = torch.mean(torch.sum(torch.abs(DVin - DVout), dim=(1,2)))

    dRhist.append(dR)
    dVhist.append(dV)

plt.plot(nskips,dRhist)
plt.ylabel("position diff")
plt.savefig("{:}{:}".format(folder,'position.png'))
plt.figure()
plt.plot(nskips,dVhist)
plt.ylabel("velocity diff")
plt.savefig("{:}{:}".format(folder,'velocity.png'))
plt.figure()
plt.plot(nskips,T[nskips])
plt.ylabel("Temperature")
plt.savefig("{:}{:}".format(folder,'temperature.png'))
plt.show()
