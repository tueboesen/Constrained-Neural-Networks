import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_npendulum_final_test/2022-08-01_12_06_30/']  # train 100, skip 20
# folders = ['/home/tue/remote_desktop/test_npendulum_table_nt10000/2022-09-09_22_18_03/']  # train 100, skip 20
folders = ['/home/tue/remote_desktop/test_npendulum_table_nt10000/2022-12-20_01_15_02/']  # train 100, skip 20

results = []
for i,folder in enumerate(folders):
    result_file = f"{folder:}results_test.npy"
    result = np.load(result_file, allow_pickle=True)
    results.append(result)
results_numpy = np.concatenate(results, axis=1)

loss_r_test = results_numpy[:,:,0].mean(axis=1)
cv_test = results_numpy[:,:,2].mean(axis=1)
cv_max_test = results_numpy[:,:,3].mean(axis=1)
mae_r_test = results_numpy[:,:,4]
mae_r_test_mean = mae_r_test.mean(axis=1)
MAE_temp = mae_r_test[:,0:2].mean(axis=1)
mae_r_test_mean_high = (mae_r_test[-1,0] + mae_r_test[-1,-1]) /2
print(f"MAEr={mae_r_test_mean*100}")
print(f"MAEr={mae_r_test_mean_high*100}")
print(f"cv = {cv_test*100}")
print("done")
