from MDAnalysis.lib.formats.libdcd import DCDFile
import numpy as np
import MDAnalysis
from MDAnalysis import *

AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
            'Y': '19','-': '20'}



# folder = '/home/tue/data/covid/'
# file_dcd = 'spike_WE.dcd'
# file_top = 'spike_WE_renumbered.psf'
folder = 'F:/TRAJECTORIES_spike_open_prot_glyc_amarolab.tar/TRAJECTORIES_spike_open_prot_glyc_amarolab/'
file_dcds = ['spike_open_prot_glyc_amarolab_1.dcd','spike_open_prot_glyc_amarolab_2.dcd','spike_open_prot_glyc_amarolab_3.dcd']
file_top = 'spike_open_prot_glyc_amarolab.psf'
file_out = 'covid_backbone'
filename_out = f"{folder}{file_out}"
filename_top=f"{folder}{file_top}"
RCAhist = []
RCBhist = []
RNhist = []
for i,file_dcd in enumerate(file_dcds):
    filename_dcd=f"{folder}{file_dcd}"
    u = Universe(filename_top, filename_dcd)

    protein = u.select_atoms("protein")
    protein_CA = protein.select_atoms("name CA")
    protein_CB = protein.select_atoms("name CB or (resname GLY and name C)")
    protein_N = protein.select_atoms("name N")

    amino_acids = protein_CA.resnames
    list_of_amino_acids = np.unique(amino_acids)
    if i==0:
        dict_AA = {k: v for v, k in enumerate(list_of_amino_acids)}
        list_of_amino_acids_org = list_of_amino_acids
        aa_num = np.asarray([dict_AA[amino_acid] for amino_acid in amino_acids])
        fragid = protein_CA.fragindices
        fragid_org = protein_CA.fragindices
        nCa = protein_CA.n_atoms
    else:
        assert (list_of_amino_acids == list_of_amino_acids_org).all()
        assert (fragid_org == fragid).all()


    nframes = u.trajectory.n_frames
    RCA = np.empty((nframes,nCa,3))
    RCB = np.empty((nframes,nCa,3))
    RN = np.empty((nframes,nCa,3))
    for j,ts in enumerate(u.trajectory):
        RCA[j] = ts.positions[protein_CA.indices]
        RCB[j] = ts.positions[protein_CB.indices]
        RN[j] = ts.positions[protein_N.indices]
    RCAhist.append(RCA)
    RCBhist.append(RCB)
    RNhist.append(RN)

RCA=np.concatenate(RCAhist,axis=0)
RCB=np.concatenate(RCBhist,axis=0)
RN=np.concatenate(RNhist,axis=0)
np.savez(filename_out,RCA=RCA,RCB=RCB,RN=RN, dict_AA=dict_AA, aa_num=aa_num,fragid=fragid)
print("all done")





