import os

import numpy as np
from src.vizualization import plot_training_and_validation_accumulated_custom, plot_pendulum_paper
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

"""
A simple file to load data from an already finished run, and plot it with custom plotting options.
 
"""
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def plot_pendulum_paper_gamma_sweep(results,legends,results_dir,colors,semilogy=False,fill_between=False,train_limits=None):
    """
    This is a more customizable version of the above function, designed for printing specific figures for papers or similar things.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['ytick.labelsize'] = 25
    mpl.rcParams['axes.labelsize'] = 25
    mpl.rcParams['legend.fontsize'] = 25
    mpl.rcParams['lines.markersize'] = 20
    mpl.rcParams['lines.markeredgewidth'] = 5
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    # mpl.rcParams['xtick.labelsize'] = 25
    # mpl.rcParams['xtick.labelsize'] = 25
    # mpl.rcParams['font.family'] = 'Arial'


    njobs, nrep, nlosses, nepochs = results.shape
    x = np.arange(nepochs)
    M = np.sum(results,axis=3) > 0

    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):
        idx = 0
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        idx = 1
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    pngfile = "{:}/Loss_r.png".format(results_dir)
    ax.set_xlim(xmin=0,xmax=150)
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()

    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):
        idx = 4
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        idx = 6
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Constraint violation (m)')
    plt.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    pngfile = "{:}/CV_training.png".format(results_dir)
    ax.set_xlim(xmin=0,xmax=150)
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    for ii in range(njobs):
        idx = 5
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii], color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii], color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

        idx = 7
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Constraint violation (m)')
    plt.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    pngfile = "{:}/CV_validation.png".format(results_dir)
    ax.set_xlim(xmin=0, xmax=150)
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    for ii in range(njobs):
        idx = 8
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii], color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii], color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

        idx = 9
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error (m)')
    plt.legend()
    # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    # ax.yaxis.set_major_locator(y_major)
    # y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # ax.yaxis.set_minor_locator(y_minor)
    # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    pngfile = "{:}/MAE_r.png".format(results_dir)
    ax.set_xlim(xmin=0, xmax=150)
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()





    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):
        idx = 4
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        idx = 5
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Mean constraint violation (m)')
    plt.legend()
    # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    # ax.yaxis.set_major_locator(y_major)
    # y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # ax.yaxis.set_minor_locator(y_minor)
    # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    pngfile = "{:}/CV.png".format(results_dir)
    ax.set_xlim(xmin=0,xmax=150)
    ax.set_ylim(ymin=0.01,ymax=0.3)
    plt.savefig(pngfile)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    for ii in range(njobs):
        idx = 6
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii], color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii], color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

        idx = 7
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, ':', color=colors[ii])
            else:
                h = ax.plot(x, y, ':', color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Max constraint violation (m)')
    plt.legend()
    # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    # ax.yaxis.set_major_locator(y_major)
    # y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    # ax.yaxis.set_minor_locator(y_minor)
    # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    pngfile = "{:}/CV_max.png".format(results_dir)
    ax.set_xlim(xmin=0, xmax=150)
    ax.set_ylim(ymin=0.05,ymax=5)
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()



    return


# folders = '/home/tue/PycharmProjects/results/test_npendulum/2022-06-25_23_38_15/' #Train1000,skip 20
folders = '/home/tue/PycharmProjects/results/test_npendulum2/2022-06-25_22_47_23/'

subfolders = get_immediate_subdirectories(folders)

# output_dir = '/home/tue/remote_desktop/regularization10/'
# os.makedirs(output_dir,exist_ok=True)
legends = [r'$\gamma=0$',r'$\gamma=100$',r'$\gamma=500$',r'$\gamma=1000$',r'$\gamma=1500$',r'$\gamma=2000$',r'$\gamma=3000$',r'$\gamma=5000$']
colors = ['black', 'orange','blue', 'red', 'green','purple','brown','grey']
repetitions = 3
epochs = 150
results=np.zeros((len(legends),repetitions,10,epochs))
for subfolder in subfolders:
    sf = subfolder.split('_')
    if sf[4] == '0':
        idx = 0
    elif sf[4] == '100':
        idx = 1
    elif sf[4] == '500':
        idx = 2
    elif sf[4] == '1000':
        idx = 3
    elif sf[4] == '1500':
        idx = 4
    elif sf[4] == '2000':
        idx = 5
    elif sf[4] == '3000':
        idx = 6
    else:
        idx = 7
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


output_dir = '/home/tue/npendulum/stab_test'
os.makedirs(output_dir,exist_ok=True)

# selected_idx = [0,1,2,7,8,13,14]
# colors = ['black', 'darkred', 'pink', 'darkgreen', 'lime', 'darkblue', 'slateblue']
# results_selected = results_numpy
# permutation = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9] #The data was ordered as: ['No constraints' 'chain high' 'chain low' 'chain reg', 'triangle high' 'triangle low' 'triangle reg' 'chaintriangle high' 'chaintriangle low' 'chaintriangle reg'], but we wish a different ordering
# results_ordered = results_numpy[permutation]
# colors = ['black', 'darkred', 'red', 'indianred', 'darkgreen', 'green', 'lime', 'darkblue', 'blue', 'slateblue']
#
# legends = ['No constraints', 'Chain', 'Triangle', 'Chaintriangle', 'End chain', 'End triangle', 'End chaintriangle', 'Reg chain', 'Reg triangle', 'Reg chaintriangle']
plot_pendulum_paper_gamma_sweep(results,legends,output_dir,colors,semilogy=True,fill_between=True,train_limits=False)
# plot_training_and_validation_accumulated_custom(result_mim,legends,output_dir_mim,colors,semilogy=True,fill_between=True,train_limits=False)



