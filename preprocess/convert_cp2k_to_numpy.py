import pandas as pd
import numpy as np

AtomicTable = {'H': 1,
     'He': 2,
     'Li': 3,
     'Be': 4,
     'B': 5,
     'C': 6,
     'S': 7,
     'O': 8
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
    # folder = '/media/tue/Data/Dropbox/ComputationalGenetics/text/Poincare_MD/MD_calculation/water_paper/'
    folder = '/home/tue/data/MD/ethanol/backup/'
    name_ener = 'ethanol-1.ener'
    name_vel = 'ethanol-vel-1.xyz'
    name_pos = 'ethanol-pos-1.xyz'
    name_force = 'forces.xyz'

    name_out = 'ethanol.npz'
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

    pos,atomic_numbers = read_xyz(filename_pos)
    vel,_ = read_xyz(filename_vel)

    force = read_xyz_force(filename_force, pos.shape[1])

    data = {'R': pos,
            'F': force,
            'V': vel,
            'z': atomic_numbers,
            'temp': temp,
            'KE': KE,
            'PE': PE
            }

    np.savez(filename_out,**data)
    # print("done")
    #
    # data2 = np.load(filename_out)
