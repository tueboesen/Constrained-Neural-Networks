import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

# folders = ['/home/tue/remote_desktop/2022-03-21_09_47_05/']
# folders = ['/home/tue/remote_desktop/2022-03-28_09_42_22/']
folders = ['/home/tue/PycharmProjects/results/test_npendulum2/2022-07-18_21_22_10/']
# folders = ['/home/tue/remote_desktop/test_MD_step_size/2022-05-19_10_29_32/']

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
results=[]
for folder in folders:
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results,axis=1)

# selected_idx = [0,3,4,5,6,9,10,11,12,15,16,17,18]
# legends = ['No constraints','Chain 1e-12','Chain 1e-4','Chain 1e-3','Chain 1e-2','Triangle 1e-12','Triangle 1e-4','Triangle 1e-3','Triangle 1e-2','ChainTriangle 1e-12','ChainTriangle 1e-4','ChainTriangle 1e-3','ChainTriangle 1e-2']
# colors = ['black', 'darkred', 'red', 'indianred', 'pink', 'darkgreen', 'green', 'lime', 'yellow', 'darkblue', 'blue', 'slateblue','purple']
# results_selected = results_numpy[selected_idx]

# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
# plot_training_and_validation_accumulated_custom(results_selected,legends,output_dir,colors,semilogy=True,fill_between=True)


output_dir = '/home/tue/remote_desktop/pendulum_reg_euler/'
output_dir = '/home/tue/remote_desktop/pendulum_reg_rk4/'
os.makedirs(output_dir,exist_ok=True)

# selected_idx = [0,1,2,7,8,13,14]
# selected_idx = [12,13,14,15,16,17,18,19,20,21,22,23]
selected_idx = [0,1,2,3,4,5,6,7,8,9,10,11]
# legends = ['No constraints','Smooth constraints','End constraints','Regularization constraints']
# colors = ['black', 'blue', 'red', 'orange']
legends = ['reg:0,pen:0','reg:0,pen:1','reg:0,pen:10','reg:0,pen:50','reg:0,pen:100','reg:0,pen:200','reg:1,pen:0','reg:1,pen:1','reg:1,pen:10','reg:1,pen:50','reg:1,pen:100','reg:1,pen:200']
colors = ['black', 'blue', 'red', 'orange','darkred','pink','lime','darkblue','slateblue','yellow','brown','green']
# colors = ['black', 'darkred', 'pink', 'darkgreen', 'lime', 'darkblue', 'slateblue']
results_selected = results_numpy[selected_idx]
# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']

plot_pendulum_paper(results_selected,legends,output_dir,colors,semilogy=True,fill_between=True,train_limits=False)

# plot_training_and_validation_accumulated_custom(results_selected,legends,output_dir,colors,semilogy=True,fill_between=True,train_limits=False)
# plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True,fill_between=True,train_limits=False)

