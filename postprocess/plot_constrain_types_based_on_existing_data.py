import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""

folders = ['E:/Dropbox/ComputationalGenetics/text/Constrained neural networks/data/2021-11-07_16_37_38/','E:/Dropbox/ComputationalGenetics/text/Constrained neural networks/data/2021-11-07_16_37_38/']
output_dir = 'E:/Dropbox/ComputationalGenetics/text/Constrained neural networks/data/'

results=[]
for folder in folders:
    result_file = f"{folder:}results.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results,axis=1)

permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
results_ordered = results_numpy[permutation]
colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_training_and_validation_accumulated_custom(results_ordered,legends,output_dir,colors,semilogy=True)

