import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import atomic_masses

AtomicTable = {'H': 1,
     'He': 2,
     'Li': 3,
     'Be': 4,
     'B': 5,
     'C': 6,
     'S': 7,
     'O': 8,
     'Ar': 18
     }

def read_xyz(filename,skiplist=None):
    with open(filename) as f:
        str = f.readline()
    n = [int(s) for s in str.split() if s.isdigit()][0]
    nskip = 2
    nmax = 600001
    if skiplist is None:
        skiplist = [i for i in range(0,nmax*(n+nskip)) if i % (n+nskip) == 0  or i % (n+nskip) == 1]
    df = pd.read_csv(filename, header=None, delimiter='\s+', skiprows=skiplist, names=['atom','x','y','z'])
    nd = df.shape[0] // n
    pos = np.empty((nd,n,3),dtype=np.float64)
    x = np.asarray(df['x'[:]]).reshape(nd,n)
    y = np.asarray(df['y'[:]]).reshape(nd,n)
    z = np.asarray(df['z'[:]]).reshape(nd,n)
    atoms = list(df['atom'][:n])
    atomic_numbers = np.asarray([AtomicTable[ele] for ele in atoms])
    pos[:,:,0] = x
    pos[:,:,1] = y
    pos[:,:,2] = z
    return pos,atomic_numbers


def read_xyz_force(filename,n):
    nskip = 5
    nmax = 600001
    skiplist = [i for i in range(0,nmax*(n+nskip)) if (i+1) % (n+nskip) == 0 or (i+1) % (n+nskip) == 1 or (i+1) % (n+nskip) == 2 or (i+1) % (n+nskip) == 3 or (i+1) % (n+nskip) == 4]
    df = pd.read_csv(filename, header=None, delimiter='\s+', skiprows=skiplist, names=['atom','kind','element','x','y','z'])
    nd = df.shape[0] // n
    pos = np.empty((nd,n,3),dtype=np.float64)
    x = np.asarray(df['x'[:]]).reshape(nd,n)
    y = np.asarray(df['y'[:]]).reshape(nd,n)
    z = np.asarray(df['z'[:]]).reshape(nd,n)
    pos[:,:,0] = x
    pos[:,:,1] = y
    pos[:,:,2] = z
    return pos




if __name__ == '__main__':
    # folder = '/media/tue/Data/Dropbox/ComputationalGenetics/text/Poincare_MD/MD_calculation/ethanol/'
    folder = '/media/tue/Data/Dropbox/ComputationalGenetics/text/Poincare_MD/MD_calculation/argon/'
    # folder = '/home/tue/data/MD/ethanol_heating/'
    name_ener = 'ar108-1.ener'
    name_vel = 'ar108-vel-1.xyz'
    name_pos = 'ar108-pos-1.xyz'
    name_force = 'forces.xyz'

    name_out = 'argon.npz'
    folder_out = folder
    # name_pos = 'test2.xyz'
    filename_ener = folder + name_ener
    filename_vel = folder + name_vel
    filename_pos = folder + name_pos
    filename_force = folder + name_force
    filename_out = folder_out + name_out

    df = pd.read_csv(filename_ener, delimiter='\s+', names=["step", 'time', 'KE', 'temp', 'PE', 'const', 'used'], skiprows=1)
    df.head()


    step = df['step'[:]]
    temp = df['temp'[:]]
    KE = df['KE'[:]]
    PE = df['PE'[:]]

    # plt.plot(temp)
    # plt.show()

    pos,atomic_numbers = read_xyz(filename_pos)
    vel,_ = read_xyz(filename_vel)
    force,_ = read_xyz(filename_force)

    # force = read_xyz_force(filename_force, pos.shape[1])

    m = atomic_masses(torch.from_numpy(atomic_numbers)).numpy()

    # CoM = np.sum(pos * m[None,:,None],axis=1,keepdims=True)/np.sum(m)
    # print("MAE CoM={:}".format(np.abs(CoM).mean()))
    #
    # pos_fixed = pos - CoM
    # CoM2 = np.sum(pos_fixed * m[None,:,None],axis=1,keepdims=True)/np.sum(m)
    # print("After fixing MAE CoM2={:}".format(np.abs(CoM2).mean()))

    #Now that Center of Mass is fixed to origo, let's rotate the coordinate system such that the first carbon atom always have y and z component 0 and such that the second carbon atom always have .


    #Lets look at angular momentum
    # L = np.cross(pos_fixed, m[None,:,None] * vel)
    # Lsum = np.sum(L,axis=1)

    #Now ensure that we are saving the same number of steps for each variable (they might have different number of data stored since we ran it while the simulation was still running)

    ndata = np.min([pos.shape[0], force.shape[0], vel.shape[0], temp.shape[0], KE.shape[0], PE.shape[0]])


    data = {'R': pos[:ndata],
            'F': force[:ndata],
            'V': vel[:ndata],
            'z': atomic_numbers,
            'temp': temp[:ndata],
            'KE': KE[:ndata],
            'PE': PE[:ndata]
            }

    np.savez(filename_out,**data)
    # print("done")
    #
    # data2 = np.load(filename_out)
