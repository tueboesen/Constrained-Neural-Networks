import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_training_and_validation_accumulated_custom_one_figure

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

output_dir_eq = '/home/tue/remote_desktop/eq/'
output_dir_mim = '/home/tue/remote_desktop/mim/'
os.makedirs(output_dir_eq,exist_ok=True)
os.makedirs(output_dir_mim,exist_ok=True)


folders = ['/home/tue/remote_desktop/test_MD_step_size/2022-05-16_14_20_30/','/home/tue/remote_desktop/test_MD_step_size/2022-05-19_10_29_32/']
# folders = ['/home/tue/remote_desktop/test_MD_step_size/2022-05-19_10_29_32/']

ntraining_samples = [1000,1000]
nsteps = [100,1000]

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
results=[]
for nt,ns,folder in zip(ntraining_samples,nsteps,folders):
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    result_eq = result[:3]
    result_mim = result[3:]
    legends = ['black', 'blue', 'red', 'black', 'blue', 'red']
    colors = ['black', 'blue', 'red', 'black', 'blue', 'red']
    fileout_eq = f"{output_dir_eq}nt{nt}_k{ns}"
    fileout_mim = f"{output_dir_mim}nt{nt}_k{ns}"
    plot_training_and_validation_accumulated_custom_one_figure(result_eq,legends,fileout_eq,colors,semilogy=True,fill_between=True,train_limits=False)
    plot_training_and_validation_accumulated_custom_one_figure(result_mim,legends,fileout_mim,colors,semilogy=True,fill_between=True,train_limits=False)

