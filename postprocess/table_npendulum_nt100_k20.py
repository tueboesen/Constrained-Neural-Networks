import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-01_12_06_30/']  # train 100, skip 20
folders = ['/home/tue/remote_desktop/test_npendulum_table_nt100/2022-09-06_09_23_02/','/home/tue/remote_desktop/test_npendulum_table_nt100/2022-09-06_21_54_27/']  # train 100, skip 20


# output = 'final_test_train100_skip100_lr1e-3'
output = 'table_nt100_k20'



results = []
for i,folder in enumerate(folders):
    result_file = f"{folder:}results_test.npy"
    result = np.load(result_file, allow_pickle=True)
    if i==0:
        result = result[:5,:2]
    results.append(result)
results_numpy = np.concatenate(results, axis=1)

loss_r_test = results_numpy[:,:,0].mean(axis=1)
cv_test = results_numpy[:,:,2].mean(axis=1)
cv_max_test = results_numpy[:,:,3].mean(axis=1)
mae_r_test = results_numpy[:,:,4]
mae_r_test_mean = mae_r_test.mean(axis=1)
mae_r_test_mean_high = (mae_r_test[-1,0] + mae_r_test[-1,-1]) /2
print(f"MAEr={mae_r_test_mean*100}")
print(f"MAEr={mae_r_test_mean_high*100}")
print(f"CV {}")
print("done")
