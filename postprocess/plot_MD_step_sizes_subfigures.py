import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_training_and_validation_accumulated_custom_one_figure

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

output_dir_eq = '/home/tue/eq/'
output_dir_mim = '/home/tue/mim/'
os.makedirs(output_dir_eq,exist_ok=True)
os.makedirs(output_dir_mim,exist_ok=True)


folders = ['/home/tue/remote_desktop/test_MD_step_size/2022-05-16_14_20_30/']
folders.append('/home/tue/remote_desktop/test_MD_step_size/2022-05-19_10_29_32/')
folders.append('/home/tue/remote_desktop/test_MD_step_size/2022-05-24_08_14_22/')
folders.append(['/home/tue/PycharmProjects/results/test_MD_step_size3/2022-05-29_15_07_41/','/home/tue/PycharmProjects/results/test_MD_step_size3/2022-05-30_09_32_58/'])
folders.append(['/home/tue/PycharmProjects/results/test_MD_step_size2/2022-05-29_15_04_25/','/home/tue/PycharmProjects/results/test_MD_step_size2/2022-05-30_09_37_59/'])
folders.append(['/home/tue/PycharmProjects/results/test_MD_step_size/2022-05-29_14_57_30/','/home/tue/PycharmProjects/results/test_MD_step_size/2022-05-30_09_29_21/'])
folders.append('/home/tue/PycharmProjects/results/test_MD_step_size2/2022-05-30_13_19_15/')
folders.append('/home/tue/PycharmProjects/results/test_MD_step_size3/2022-05-30_13_30_29/')
folders.append('/home/tue/remote_desktop/test_MD_step_size/2022-05-30_09_53_10/')
# folders = ['/home/tue/remote_desktop/test_MD_step_size/2022-05-19_10_29_32/']

ntraining_samples = [1000,1000,10000,100,100,100,100,1000,1000]
nsteps = [100,1000,1000,10,1000,100,10000,10000,10]

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
for nt,ns,folder in zip(ntraining_samples,nsteps,folders):
    if isinstance(folder,list):
        results = []
        for subfolder in folder:
            result_file = f"{subfolder:}results.npy"
            result = np.load(result_file, allow_pickle=True)
            results.append(result)
        results = np.concatenate(results, axis=1)
    else:
        result_file = f"{folder:}results.npy"
        results = np.load(result_file, allow_pickle=True)
    result_eq = results[:3]
    result_mim = results[3:]
    legends = ['black', 'blue', 'red', 'black', 'blue', 'red']
    colors = ['black', 'blue', 'red', 'black', 'blue', 'red']
    fileout_eq = f"{output_dir_eq}nt{nt}_k{ns}"
    fileout_mim = f"{output_dir_mim}nt{nt}_k{ns}"
    plot_training_and_validation_accumulated_custom_one_figure(result_eq,legends,fileout_eq,colors,semilogy=True,fill_between=True,train_limits=False)
    plot_training_and_validation_accumulated_custom_one_figure(result_mim,legends,fileout_mim,colors,semilogy=True,fill_between=True,train_limits=False)

