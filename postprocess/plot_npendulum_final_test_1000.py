import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-01_12_06_30/']  # train 100, skip 20
folders = ['/home/tue/PycharmProjects/results/test_npendulum_final_test_1000/2022-08-03_12_16_42/']  # train 100, skip 20


# output = 'final_test_train100_skip100_lr1e-3'
output = 'final_test_train1000_skip100_lr_var'

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
results = []
for folder in folders:
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results, axis=1)
results_numpy = np.delete(results_numpy,3,1)
# results_numpy = np.delete(results_numpy,2,1)
# results_numpy = np.delete(results_numpy,1,1)
# results_numpy = np.delete(results_numpy,0,1)
# results_numpy = results_numpy[:,:]

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


base_output_dir = '/home/tue/npendulum_figures/'
output_dir = f"{base_output_dir}{output}"
os.makedirs(output_dir, exist_ok=True)

# selected_idx = [0,1,2,7,8,13,14]
# legends = ['No constraints', 'Stabilized', 'End constraints + Reg + Stabilized', 'Smooth constraints + Reg + Stabilized']
legends = ['No constraints', r'$\gamma = 50$', r'End constraints, $ \gamma = 50, \eta = 5$', r'Smooth constraints, $ \gamma = 50, \eta = 5$']
colors = ['black','orange', 'blue', 'red']
# colors = ['black', 'darkred', 'pink', 'darkgreen', 'lime', 'darkblue', 'slateblue']
# results_selected = results_numpy
# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_pendulum_paper(results_numpy, legends, output_dir, colors, semilogy=True, fill_between=True, train_limits=False)
# plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True,fill_between=True,train_limits=False)

