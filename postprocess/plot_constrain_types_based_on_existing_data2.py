import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

folders = ['/home/tue/remote_desktop/2022-03-15_10_26_34/']
output_dir = '/home/tue/remote_desktop/regularization/'
os.makedirs(output_dir,exist_ok=True)
results=[]
for folder in folders:
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results,axis=1)

selected_idx = [0,3,4,5,6,9,10,11,12,15,16,17,18]
legends = ['No constraints','Chain 1e-12','Chain 1e-4','Chain 1e-3','Chain 1e-2','Triangle 1e-12','Triangle 1e-4','Triangle 1e-3','Triangle 1e-2','ChainTriangle 1e-12','ChainTriangle 1e-4','ChainTriangle 1e-3','ChainTriangle 1e-2']
colors = ['black', 'darkred', 'red', 'indianred', 'pink', 'darkgreen', 'green', 'lime', 'yellow', 'darkblue', 'blue', 'slateblue','purple']
results_selected = results_numpy[selected_idx]

# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_training_and_validation_accumulated_custom(results_selected,legends,output_dir,colors,semilogy=True,fill_between=True)


output_dir = '/home/tue/remote_desktop/inherent/'
os.makedirs(output_dir,exist_ok=True)


selected_idx = [0,1,2,7,8,13,14]
legends = ['No constraints','Chain','Chain end','Triangle','Triangle end','ChainTriangle','ChainTriangle end']
colors = ['black', 'darkred', 'pink', 'darkgreen', 'lime', 'darkblue', 'slateblue']
results_selected = results_numpy[selected_idx]
# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_training_and_validation_accumulated_custom(results_selected,legends,output_dir,colors,semilogy=True,fill_between=True)

