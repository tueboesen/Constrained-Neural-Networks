import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

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

    fig, ax = plt.subplots(num=1,figsize=(15,15), clear=True)
    for ii in range(njobs):

        idx = 0
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
            ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/Loss_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_training.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
    fig, ax = plt.subplots(num=1,figsize=(15,15), clear=True)
    for ii in range(njobs):
        idx = 1
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/Loss_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()

    fig, ax = plt.subplots(num=1, figsize=(15,15), clear=True)
    for ii in range(njobs):

        idx = 2
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('LossD')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/LossD_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/LossD_training.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
    fig, ax = plt.subplots(num=1, figsize=(15,15), clear=True)

    for ii in range(njobs):
        idx = 3
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
            else:
                h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('LossD')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/LossD_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/LossD_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.clf()
    return


def plot_training_and_validation_accumulated_custom(results,legends,results_dir,colors,semilogy=False,fill_between=False):
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

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # ax.ticklabel_format(style='plain')
    # plt.yticks(fontsize=25)

    # ax.set_xticks([0,100,200,300])
    plt.legend()
    # plt.title("Training")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.set_yticks([20, 200, 500])
    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if semilogy:
        pngfile = "{:}/Loss_{:}_training.png".format(results_dir, 'semilogy')
    else:
        pngfile = "{:}/Loss_training.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):
        idx = 1
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii],color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii],color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.title("Training")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if semilogy:
        pngfile = "{:}/Loss_{:}_validation.png".format(results_dir, 'semilogy')
    else:
        pngfile = "{:}/Loss_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
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
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('LossD')
    plt.legend()
    # plt.title("Training")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if semilogy:
        pngfile = "{:}/LossD_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/LossD_training.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
    fig, ax = plt.subplots(figsize=(15,15))
    for ii in range(njobs):
        idx = 3
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '-', label=legends[ii], color=colors[ii])
            else:
                h = ax.plot(x, y, '-', label=legends[ii], color=colors[ii])
            if fill_between:
                ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('LossD')
    plt.legend()
    # plt.title("Validation")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    y_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if semilogy:
        pngfile = "{:}/LossD_{:}_validation.png".format(results_dir, 'semilogy')
    else:
        pngfile = "{:}/LossD_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.close()
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