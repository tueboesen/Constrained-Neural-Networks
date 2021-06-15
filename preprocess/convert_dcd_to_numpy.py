from MDAnalysis.lib.formats.libdcd import DCDFile
import numpy as np
import MDAnalysis
from MDAnalysis import *

AA_DICT = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
            'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18',
            'Y': '19','-': '20'}



folder = '/home/tue/data/covid/'
file_dcd = 'spike_WE.dcd'
file_top = 'spike_WE_renumbered.psf'
file_out = 'covid_backbone'
filename_dcd=f"{folder}{file_dcd}"
filename_top=f"{folder}{file_top}"
filename_out = f"{folder}{file_out}"
u = Universe(filename_top, filename_dcd)

protein = u.select_atoms("protein")
protein_CA = protein.select_atoms("name CA")
protein_CB = protein.select_atoms("name CB or (resname GLY and name C)")
protein_N = protein.select_atoms("name N")

amino_acids = protein_CA.resnames
list_of_amino_acids = np.unique(amino_acids)
dict_AA = {k: v for v, k in enumerate(list_of_amino_acids)}
aa_num = np.asarray([dict_AA[amino_acid] for amino_acid in amino_acids])
fragid = protein_CA.fragindices

nCa = protein_CA.n_atoms

nframes = u.trajectory.n_frames
RCA = np.empty((nframes,nCa,3))
RCB = np.empty((nframes,nCa,3))
RN = np.empty((nframes,nCa,3))
for i,ts in enumerate(u.trajectory):
    RCA[i] = ts.positions[protein_CA.indices]
    RCB[i] = ts.positions[protein_CB.indices]
    RN[i] = ts.positions[protein_N.indices]

np.savez(filename_out,RCA=RCA,RCB=RCB,RN=RN, dict_AA=dict_AA, aa_num=aa_num,fragid=fragid)
print("all done")





