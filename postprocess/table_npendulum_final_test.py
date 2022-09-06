import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-01_12_06_30/']  # train 100, skip 20
folders = ['/home/tue/PycharmProjects/results/test_npendulum_table_nt100/2022-09-06_08_56_35/']  # train 100, skip 20


results = []
for folder in folders:
    result_file = f"{folder:}results_test.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results, axis=1)

loss_r_test = results[0][:,:,0].mean(axis=1)
cv_test = results[0][:,:,2].mean(axis=1)
cv_max_test = results[0][:,:,3].mean(axis=1)
mae_r_test = results[0][:,:,4].mean(axis=1)
print(f"MAEr={mae_r_test}")
