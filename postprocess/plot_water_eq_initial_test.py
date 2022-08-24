import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper, plot_water_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_water/2022-08-15_15_34_01/']  # train 100, skip 20
folders = ['/home/tue/PycharmProjects/results/test_water/2022-08-19_13_48_27/']  # train 1000, skip 50
folders = ['/home/tue/PycharmProjects/results/test_water_100/2022-08-23_09_10_33/']  # train 100, skip 50


output = 'eq_initial_test_100'

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
results = []
for folder in folders:
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results, axis=1)



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
Rscale = 21.0823
result[:,:,4:8] = result[:,:,4:8] / Rscale

MAEtrainidx = 8
MAEtestidx = 9
MAE_train = result[:,:,MAEtrainidx,:].min(axis=-1).mean(axis=1)*100
MAE_test = result[:,:,MAEtestidx,:].min(axis=-1).mean(axis=1)*100

MAE_train_std = result[:,:,MAEtrainidx,:].min(axis=-1).std(axis=1)*100
MAE_test_std = result[:,:,MAEtestidx,:].min(axis=-1).std(axis=1)*100


base_output_dir = '/home/tue/water_figures/'
output_dir = f"{base_output_dir}{output}"
os.makedirs(output_dir, exist_ok=True)

# selected_idx = [0,1,2,7,8,13,14]
legends = ['No constraints', r'$\gamma = 10$', r'End constraints, $ \gamma = 10, \eta = 5000$', r'Smooth constraints, $ \gamma = 10, \eta = 5000$']
colors = ['black','orange', 'blue', 'red']


plot_water_paper(result, legends, output_dir, colors, semilogy=True, fill_between=True, train_limits=False)
# plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True,fill_between=True,train_limits=False)

