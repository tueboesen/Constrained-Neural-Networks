import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

folders = ['E:/Dropbox/ComputationalGenetics/text/Constrained neural networks/data/2021-11-07_16_37_38/','E:/Dropbox/ComputationalGenetics/text/Constrained neural networks/data/2021-11-07_16_37_38/']
output_dir = '/home/tue/remote_desktop/'
output_dir_eq = '/home/tue/remote_desktop/eq'
output_dir_mim = '/home/tue/remote_desktop/mim'
os.makedirs(output_dir_mim,exist_ok=True)
os.makedirs(output_dir_eq,exist_ok=True)

result_file = '/home/tue/remote_desktop/results.npy'

# results[c['jobid'], c['repetition'], 0, :] = result['loss_t']
# results[c['jobid'], c['repetition'], 1, :] = result['loss_v']
# results[c['jobid'], c['repetition'], 2, :] = result['lossD_t']
# results[c['jobid'], c['repetition'], 3, :] = result['lossD_v']

result = np.load(result_file, allow_pickle=True)

result_eq = result[:2]
result_mim = result[2:]

# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
colors = ['black', 'red']
legends = ['No constraints', 'Triangle']
plot_training_and_validation_accumulated_custom(result_eq,legends,output_dir_eq,colors,semilogy=True)

colors = ['black', 'red']
legends = ['No constraints', 'Triangle']
plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True)



colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_training_and_validation_accumulated_custom(result,legends,output_dir,colors,semilogy=True)

