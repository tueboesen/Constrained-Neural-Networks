import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.

"""

# folders = ['/home/tue/PycharmProjects/results/test_water_100/2022-08-23_09_10_33/']
folders = ['/home/tue/results/test_water_100/2022-12-19_00_42_28/']

#NOTE that this one does not have the correct ones for the projections, those are in a separate folder


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
# mae_r_test_mean_high = (mae_r_test[-1,0] + mae_r_test[-1,-1]) /2
print(f"MAEr={mae_r_test_mean*100}")
print(f"cv = {cv_test*100}")
# print(f"MAEr={mae_r_test_mean_high*100}")
print("done")