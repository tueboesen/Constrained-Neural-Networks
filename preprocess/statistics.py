import numpy as np
import matplotlib.pyplot as plt
from src.utils import convert_snapshots_to_future_state_dataset, Distogram
import torch

folder = '/home/tue/data/MD/ethanol/backup2/'
name_ener = 'ethanol-1.ener'
name_vel = 'ethanol-vel-1.xyz'
name_pos = 'ethanol-pos-1.xyz'
name_force = 'forces.xyz'

name_out = 'ethanol_70k.npz'
folder_out = folder
filename_out = folder_out + name_out

data = np.load(filename_out)

nskips = torch.linspace(0,50000,100).round().to(dtype=torch.int64)
R = torch.from_numpy(data['R'])
V = torch.from_numpy(data['V'])


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
    # dV = torch.mean(torch.norm(Vin - Vout, p=2, dim=(1,2)))
    # MAEr = torch.mean(torch.abs(Rin - Rout))

    dRhist.append(dR)
    dVhist.append(dV)
    # MAErhist.append(MAEr)

plt.plot(nskips,dRhist)
plt.ylabel("position diff")
plt.figure()
plt.plot(nskips,dVhist)
plt.ylabel("velocity diff")
# plt.axi
plt.show()
print("done")