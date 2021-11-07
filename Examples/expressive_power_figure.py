import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np

# glob("/path/to/directory/*/")
#
# npzfiles = [f for f in glob.glob(search_command)]
legends=['No constraints','Constraints']
path = 'E:/Dropbox/ComputationalGenetics/text/Poincare_MD/Only constraints/figures/expressive power2/2021-09-01_12_40_38/'
subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
results = []
counter = 0
for subfolder in subfolders:
    if subfolder[-1] == '0':
        results.append([])
    file = subfolder + '/training.csv'
    tmp = pd.read_csv(file,sep='\t')
    results[-1].append(np.asarray(tmp['loss_t']))
    x = np.asarray(tmp['epoch'])


y=np.asarray(results)

y_mean = np.mean(y,axis=1)
y_std = np.std(y, axis=1)
for i in range(y.shape[0]):
    h = plt.plot(x, y_mean[i], '-o', label=legends[i], markersize=3)
    # plt.fill_between(x, y_mean[i] - y_std[i], y_mean[i] + y_std[i], color=h[0].get_color(), alpha=0.2)

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss')
plt.legend()
pngfile = "{:}/expressive_power_loss.png".format(path)
plt.savefig(pngfile)


#
# files = ['training0.csv','training1.csv']
# legends = ['No constraints','Chain']
# data = []
# pngfile = "{:}/expressive_power_loss.png".format(path)
# fig, ax = plt.subplots()
#
# for file,legend in zip(files,legends):
#     filepath = path + file
#     tmp = pd.read_csv(filepath,sep='\t')
#     loss = tmp['loss_t']
#     x = tmp['epoch']
#     data.append(loss)
#
#     ax.plot(x, loss, '-',label=legend)
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig(pngfile)
# print("done")
