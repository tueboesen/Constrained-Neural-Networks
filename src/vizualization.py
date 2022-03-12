import matplotlib.pyplot as plt
import numpy as np

def plot_training_and_validation_accumulated(results,legends,results_dir,semilogy=False):
    """
    plots the training and validation data as it accumulates over several jobs in a big run.
    Expects the results to be a numpy variable, with shape (ntypes,nreps,nlosses,nepochs)
      nlosses should be 4 and should contain the following: loss_t,loss_v,lossD_t,lossD_v (in that order)
    Note that this function works even if not all the data is currently in the results.
    It will only plot data where at least one datapoint is different from zero, and is even aware of the number of repetitions current filled out in results and will take that into account when plotting mean and std.
    """
    njobs, nrep, nlosses, nepochs = results.shape
    x = np.arange(nepochs)
    M = np.sum(results,axis=3) > 0
    if semilogy:
        pngfile = "{:}/Loss_{:}.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss.png".format(results_dir)

    fig, ax = plt.subplots(num=1, clear=True)
    for ii in range(njobs):

        idx = 0
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}-training")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}-training")
            ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        idx = 1
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}-validation")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}-validation")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    plt.savefig(pngfile)
    plt.clf()

    if semilogy:
        pngfile = "{:}/LossD_{:}.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/LossD.png".format(results_dir)
    fig, ax = plt.subplots(num=1, clear=True)
    for ii in range(njobs):

        idx = 2
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}-training")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}-training")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

        idx = 3
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}-validation")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}-validation")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel('LossD')
        plt.legend()
    plt.savefig(pngfile)
    plt.clf()
    return


def plot_training_and_validation_accumulated_custom(results,legends,results_dir,colors,semilogy=False):
    """
    This is a more customizable version of the above function, designed for printing specific figures for papers or similar things.
    """
    njobs, nrep, nlosses, nepochs = results.shape
    x = np.arange(nepochs)
    M = np.sum(results,axis=3) > 0
    if semilogy:
        pngfile = "{:}/Loss_{:}.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss.png".format(results_dir)

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
            ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        idx = 1
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    plt.savefig(pngfile)
    plt.clf()

    if semilogy:
        pngfile = "{:}/LossD_{:}.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/LossD.png".format(results_dir)
    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):

        idx = 2
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

        idx = 3
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii], color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii], color=colors[ii])
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel('LossD')
        plt.legend()
    plt.savefig(pngfile)
    plt.clf()
    return


def plot_training_and_validation(results,result_dir):
    """
    Plots the training and validation results of a single run.
    """
    fig = plt.figure(num=1)
    plt.clf()
    plt.plot(results['epoch'], results['loss_t'], label='training')
    plt.plot(results['epoch'], results['loss_v'], label='validation')
    plt.ylim(0, 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(f"{result_dir}/loss.png")

    plt.clf()
    plt.plot(results['epoch'], results['lossD_t'], label='training')
    plt.plot(results['epoch'], results['lossD_v'], label='validation')
    plt.ylim(0, 1)
    plt.xlabel('epochs')
    plt.ylabel('lossD')
    plt.legend()
    # plt.show()
    plt.savefig(f"{result_dir}/lossD.png")
    return

if __name__ == '__main__':
    file = ''