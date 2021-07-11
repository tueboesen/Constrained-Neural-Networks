import torch
import numpy as np
import glob
import time


folder = '/home/tue/data/casp11/npz/'
search_command = folder + "*.npz"
npzfiles = [f for f in glob.glob(search_command)]
output_file = '/home/tue/data/casp11/protein_stat.npz'

dict_AA = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HSD': 8, 'HSE': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20}
# dict_AA_short = {'A':0, 'C':4, 'D':3, 'E':6, 'F':14, 'G':7, 'H':8, 'I':10, 'K':12, 'L':11, 'M':13, 'N':2, 'P':15,'Q':5, 'R':1, 'S':16, 'T':17, 'V':20, 'W':18, 'Y':19, '-'} #Note that H should cover both 8 and 9, we will just have to copy info from one to the other afterwards.
dict_AA_short = {'A':0, 'C':4, 'D':3, 'E':6, 'F':13, 'G':7, 'H':8, 'I':9, 'K':11, 'L':10, 'M':12, 'N':2, 'P':14,'Q':5, 'R':1, 'S':15, 'T':16, 'V':19, 'W':17, 'Y':18} #Note that H should cover HSE and HSD, we will just have to copy info from one to the other afterwards.
log_units = -9

n = len(dict_AA_short)
dnn = np.zeros((n,n))
cnn = np.zeros((n,n),dtype=np.int64)

dab = np.zeros(n)
dan = np.zeros(n)
dbn = np.zeros(n)
cab = np.zeros((n),dtype=np.int64)
can = np.zeros((n),dtype=np.int64)
cbn = np.zeros((n),dtype=np.int64)
t0 = time.time()

for i,npzfile in enumerate(npzfiles):
    if (i+1) % 100==0:
        print(f"{i}, time={time.time() - t0:2.2f}s")
    data = np.load(npzfile)
    rCa = data['rCa']
    rCb = data['rCb']
    rN = data['rN']
    AA_list = data['AA_LIST']
    seq = data['seq']
    M1 = rCa != 0
    M1 = np.min(M1,axis=0)
    M2 = rCb != 0
    M2 = np.min(M2,axis=0)
    M3 = rN != 0
    M3 = np.min(M3,axis=0)
    M = M1 * M2 * M3
    Mnn = M[1:] * M[:-1]
    #Now we need to find the correct chain distance and the correct amino acid distances

    seq_letters = AA_list[seq]
    seq_conv = [dict_AA_short[seq_letter] for seq_letter in seq_letters]
    sc = np.asarray(seq_conv)

    dA = np.linalg.norm(rCa[:,1:] - rCa[:,:-1],axis=0)
    dnn[sc[:-1][Mnn],sc[1:][Mnn]] += dA[Mnn]
    cnn[sc[:-1][Mnn],sc[1:][Mnn]] += 1

    #Next we find the correct distances within an amino acid
    dAB = np.linalg.norm(rCa - rCb,axis=0)
    dAN = np.linalg.norm(rCa - rN,axis=0)
    dBN = np.linalg.norm(rCb - rN,axis=0)

    dab[seq[M]] += dAB[M]
    dan[seq[M]] += dAN[M]
    dbn[seq[M]] += dBN[M]
    cab[seq[M]] += 1
    can[seq[M]] += 1
    cbn[seq[M]] += 1

dnn_mean = dnn / cnn
dab_mean = dab / cab
dan_mean = dan / can
dbn_mean = dbn / cbn

np.savez(output_file,dnn=dnn_mean,dab=dab_mean,dan=dan_mean,dbn=dbn_mean,dict_AA=dict_AA_short,log_units=log_units)
print("done")


