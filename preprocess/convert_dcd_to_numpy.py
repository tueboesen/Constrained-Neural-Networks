from MDAnalysis.lib.formats.libdcd import DCDFile
import numpy as np
import MDAnalysis
from MDAnalysis import *

folder = '/home/tue/data/covid/'
file_dcd = 'spike_WE.dcd'
file_top = 'spike_WE_renumbered.psf'
filename_dcd=f"{folder}{file_dcd}"
filename_top=f"{folder}{file_top}"
u = Universe(filename_top, filename_dcd)
p = u.select_atoms("protein")
p = u.select_atoms("backbone")
print("done")








# with DCDFile(filename_dcd) as dcd:
#     header = dcd.header
#     # iterate over trajectory
#     print('length of dcd {:}'.format(len(dcd)))
#     xyz = np.empty((len(dcd),header['natoms'], 3))
#     for i,frame in enumerate(dcd):
#         xyz[i] = frame.xyz


with MDAnalysis.topology.TOPParser.TOPParser(filename_top) as top:
    print("niw?")
print("done")