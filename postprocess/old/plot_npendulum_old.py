import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper
import csv

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# folders = '/home/tue/PycharmProjects/results/test_npendulum/2022-06-25_23_38_15/' #Train1000,skip 20
folders = '/home/tue/PycharmProjects/results/test_npendulum2/2022-06-25_22_47_23/'

subfolders = get_immediate_subdirectories(folders)

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
legends = ['Stabilized','Smooth constraints','End constraints','No constraints']
colors = ['orange', 'blue', 'red', 'black']
repetitions = 3
epochs = 150
results=np.zeros((len(legends),repetitions,10,epochs))
for subfolder in subfolders:
    sf = subfolder.split('_')
    if sf[2] == 'stabhigh':
        idx = 0
    elif sf[2] == 'high':
        idx = 1
    elif sf[2] == 'low':
        idx = 2
    else:
        idx = 3
    rep = int(sf[-1])
    result_file = f"{folders}{subfolder}/training.csv"
    with open(result_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for i,row in enumerate(reader):
            if i == 0:
                continue
            results[idx,rep,0,i-1] = float(row[2])
            results[idx,rep,1,i-1] = float(row[4])
            results[idx,rep,2,i-1] = float(row[3])
            results[idx,rep,3,i-1] = float(row[5])
            results[idx,rep,4,i-1] = float(row[3])
            results[idx,rep,5,i-1] = float(row[5])
            results[idx,rep,6,i-1] = float(row[6])
            results[idx,rep,7,i-1] = float(row[7])
            results[idx,rep,8,i-1] = float(row[8])
            results[idx,rep,9,i-1] = float(row[9])

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


output_dir = '/home/tue/npendulum/'
os.makedirs(output_dir,exist_ok=True)

# selected_idx = [0,1,2,7,8,13,14]
# colors = ['black', 'darkred', 'pink', 'darkgreen', 'lime', 'darkblue', 'slateblue']
# results_selected = results_numpy
# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_pendulum_paper(results,legends,output_dir,colors,semilogy=True,fill_between=True,train_limits=False)
# plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True,fill_between=True,train_limits=False)

