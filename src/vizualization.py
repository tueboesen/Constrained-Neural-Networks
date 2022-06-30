import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits import mplot3d
import torch

from colour import Color

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plot_pendulum_snapshots(Rin,Routs,Vin,Vouts,Rpred=None,Vpred=None,file=None):
    n = Rin.shape[0]
    m = Routs.shape[0]
    fig, ax = plt.subplots(figsize=(15,15))
    origo = torch.tensor([[0.0,0.0]])

    Rin = torch.cat((origo,Rin),dim=0)
    Routs = torch.cat((origo[None,:,:].repeat(m,1,1),Routs),dim=1)
    if Rpred is not None:
        Rpred = torch.cat((origo,Rpred),dim=0)

    Rin = Rin.numpy()
    Vin = Vin.numpy()
    Routs = Routs.numpy()
    Vouts = Vouts.numpy()

    v_origo =np.asarray([Rin[1:,0], Rin[1:,1]])
    # plt.quiver(*v_origo, Vin[:, 0], Vin[:, 1], color='r', scale=200,width=0.003, alpha=0.2)

    # l_in, = ax.plot(Rin[:,0], Rin[:,1],color='pink', alpha=0.1)
    # lm_in, = ax.plot(Rin[:,0], Rin[:,1], 'ro',  alpha=0.5)

    # colors = list(red.range_to(Color("blue"), m))
    c1 = 'red'
    c2 = 'blue'


    for i in range(m):
        for j in range(0,n+1):
            Rout = Routs[i,j]
            # Vout = Vouts[i,j]
            c = colorFader(c1, c2, j / n)
            # v_origo =np.asarray([Rout[1:,0], Rout[1:,1]])
            # plt.quiver(*v_origo, Vout[:, 0], Vout[:, 1], color=c, scale=200,width=0.003,alpha=0.2)

            l_out, = ax.plot(Routs[i,:,0], Routs[i,:,1],color='gray',  alpha=0.03)
            lm_out, = ax.plot(Rout[0], Rout[1], 'o', color=c,  alpha=0.5)

    if Rpred is not None:
        l_p, = ax.plot(Rpred[:,0], Rpred[:,1], '--', color='lime', alpha=0.7)
        lm_p, = ax.plot(Rpred[:,0], Rpred[:,1], 'go', alpha=0.7)

    # ax.set_xlim(-n,n)
    # ax.set_ylim(-n,n)
    # plt.axis('scaled')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    if file is None:
        plt.show()
        plt.pause(1)
    else:
        os.makedirs(os.path.dirname(file),exist_ok=True)
        plt.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close()
    return



def plot_pendulum_snapshot(Rin,Rout,Vin,Vout,Rpred=None,Vpred=None,file=None):
    n = Rin.shape[0]
    fig, ax = plt.subplots(figsize=(15,15))
    origo = torch.tensor([[0.0,0.0]])

    Rin = torch.cat((origo,Rin),dim=0)
    Rout = torch.cat((origo,Rout),dim=0)
    if Rpred is not None:
        Rpred = torch.cat((origo,Rpred),dim=0)

    Rin = Rin.numpy()
    Vin = Vin.numpy()
    Rout = Rout.numpy()
    Vout = Vout.numpy()

    v_origo =np.asarray([Rin[1:,0], Rin[1:,1]])
    plt.quiver(*v_origo, Vin[:, 0], Vin[:, 1], color='r', scale=200,width=0.003)

    v_origo =np.asarray([Rout[1:,0], Rout[1:,1]])
    plt.quiver(*v_origo, Vout[:, 0], Vout[:, 1], color='b', scale=200,width=0.003)


    l_in, = ax.plot(Rin[:,0], Rin[:,1],color='pink', alpha=0.7)
    lm_in, = ax.plot(Rin[:,0], Rin[:,1], 'ro',  alpha=0.7)

    l_out, = ax.plot(Rout[:,0], Rout[:,1],color='lightblue',  alpha=0.7)
    lm_out, = ax.plot(Rout[:,0], Rout[:,1], 'bo',  alpha=0.7)

    if Rpred is not None:
        l_p, = ax.plot(Rpred[:,0], Rpred[:,1], '--', color='lime', alpha=0.7)
        lm_p, = ax.plot(Rpred[:,0], Rpred[:,1], 'go', alpha=0.7)

    ax.set_xlim(-n,n)
    ax.set_ylim(-n,n)
    if file is None:
        plt.show()
        plt.pause(1)
    else:
        os.makedirs(os.path.dirname(file),exist_ok=True)
        plt.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close()
    return



def plot_water(r_new,v_new,r_old,v_old,r_org,v_org):
    mpl.use('TkAgg')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    r_new = r_new[0]
    r_old = r_old[0]
    r_org = r_org[0]
    n_mol = 5
    ax.scatter3D(r_new[:n_mol,0],r_new[:n_mol,1],r_new[:n_mol,2],color='red',s=100)
    ax.scatter3D(r_old[:n_mol,0],r_old[:n_mol,1],r_old[:n_mol,2],color='pink',s=100)
    ax.scatter3D(r_org[:n_mol,0],r_org[:n_mol,1],r_org[:n_mol,2],color='black',s=100)

    ax.scatter3D(r_new[:n_mol,3],r_new[:n_mol,4],r_new[:n_mol,5],color='blue',s=50)
    ax.scatter3D(r_old[:n_mol,3],r_old[:n_mol,4],r_old[:n_mol,5],color='darkblue',s=50)
    ax.scatter3D(r_org[:n_mol,3],r_org[:n_mol,4],r_org[:n_mol,5],color='black',s=50)

    ax.scatter3D(r_new[:n_mol,6],r_new[:n_mol,7],r_new[:n_mol,8],color='green',s=50)
    ax.scatter3D(r_old[:n_mol,6],r_old[:n_mol,7],r_old[:n_mol,8],color='darkgreen',s=50)
    ax.scatter3D(r_org[:n_mol,6],r_org[:n_mol,7],r_org[:n_mol,8],color='black',s=50)
    assert np.max(np.abs(v_new-v_old)) < 1e-10

    plt.show()
    plt.pause(1)
    print("done")


def plot_training_and_validation_accumulated_2(results,legends,results_dir,semilogy=False):
    """
    plots the training and validation data as it accumulates over several jobs in a big run.
    Expects the results to be a numpy variable, with shape (ntypes,nreps,nlosses,nepochs)
    nlosses should be 4 and should contain the following: loss_r_t,loss_r_v,loss_v_t,loss_v_v (in that order)
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
    plt.ylabel('Loss_r')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/Loss_r_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_r_training.png".format(results_dir)
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
    plt.ylabel('Loss_r')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/Loss_r_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_r_validation.png".format(results_dir)
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
    plt.ylabel('Loss_v')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/Loss_v_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_v_training.png".format(results_dir)
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
    plt.ylabel('Loss_v')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/Loss_v_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_v_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.clf()
    return



def plot_training_and_validation_accumulated_4(results,legends,results_dir,semilogy=False):
    """
    plots the training and validation data as it accumulates over several jobs in a big run.
    Expects the results to be a numpy variable, with shape (ntypes,nreps,nlosses,nepochs)
    nlosses should be 4 and should contain the following: loss_r_t,loss_r_v,loss_v_t,loss_v_v (in that order)
    Note that this function works even if not all the data is currently in the results.
    It will only plot data where at least one datapoint is different from zero, and is even aware of the number of repetitions current filled out in results and will take that into account when plotting mean and std.
    """
    njobs, nrep, nlosses, nepochs = results.shape
    x = np.arange(nepochs)
    M = np.sum(results,axis=3) > 0
    ylabels = ['Loss_r','Loss_v','cv','cv_max','MAE_r']

    for kk,ylabel in enumerate(ylabels):
        fig, ax = plt.subplots(num=1, figsize=(15, 15), clear=True)
        for ii in range(njobs):
            idx = 2*kk
            if np.sum(M[ii, :, idx]) > 0:
                y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
                ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
                if semilogy:
                    h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
                else:
                    h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.title("Training")
        pngfile = f"{results_dir}/{ylabel}_{'semilogy' if semilogy else ''}_training.png"
        plt.savefig(pngfile)
        plt.close()
        fig, ax = plt.subplots(num=1, figsize=(15, 15), clear=True)
        for ii in range(njobs):
            idx = 2*kk+1
            if np.sum(M[ii, :, idx]) > 0:
                y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
                ystd = results[ii,M[ii,:,idx],idx,:].std(axis=0)
                if semilogy:
                    h = ax.semilogy(x, y, '-', label=f"{legends[ii]:}")
                else:
                    h = ax.plot(x, y, '-', label=f"{legends[ii]:}")
                ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.title("Validation")
        pngfile = f"{results_dir}/{ylabel}_{'semilogy' if semilogy else ''}_validation.png"
        plt.savefig(pngfile)
        plt.close()
    return



def plot_training_and_validation_accumulated_3(results,legends,results_dir,semilogy=False):
    """
    plots the training and validation data as it accumulates over several jobs in a big run.
    Expects the results to be a numpy variable, with shape (ntypes,nreps,nlosses,nepochs)
    nlosses should be 4 and should contain the following: loss_r_t,loss_r_v,loss_v_t,loss_v_v (in that order)
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
    plt.ylabel('Loss_r')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/Loss_r_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_r_training.png".format(results_dir)
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
    plt.ylabel('Loss_r')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/Loss_r_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_r_validation.png".format(results_dir)
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

        idx = 4
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '--', color=h[0].get_color())
            else:
                h = ax.plot(x, y, '--', color=h[0].get_color())
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)


    plt.xlabel('Epochs')
    plt.ylabel('Loss_v')
    plt.legend()
    plt.title("Training")
    if semilogy:
        pngfile = "{:}/Loss_v_{:}_training.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_v_training.png".format(results_dir)
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

        idx = 5
        if np.sum(M[ii, :, idx]) > 0:
            y = results[ii, M[ii, :, idx], idx, :].mean(axis=0)
            ystd = results[ii, M[ii, :, idx], idx, :].std(axis=0)
            if semilogy:
                h = ax.semilogy(x, y, '--', color=h[0].get_color())
            else:
                h = ax.plot(x, y, '--', color=h[0].get_color())
            ax.fill_between(x, y - ystd, y + ystd, color=h[0].get_color(), alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss_v')
    plt.legend()
    plt.title("Validation")
    if semilogy:
        pngfile = "{:}/Loss_v_{:}_validation.png".format(results_dir,'semilogy')
    else:
        pngfile = "{:}/Loss_v_validation.png".format(results_dir)
    plt.savefig(pngfile)
    plt.clf()
    return



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


def plot_training_and_validation_accumulated_custom(results,legends,results_dir,colors,semilogy=False,fill_between=False,train_limits=None):
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
    ax.set_xlim(xmin=0,xmax=250)
    ax.set_ylim(ymax=0.3)
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
    ax.set_xlim(xmin=0,xmax=250)
    ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    # plt.savefig(pngfile,bbox_inches="tight", pad_inches=0)
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
    if train_limits:
        plt.ylim(0.07,0.3)
        # ax.set_ylim(0.07,0.3)

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


def plot_pendulum_paper(results,legends,results_dir,colors,semilogy=False,fill_between=False,train_limits=None):
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
    # ax.set_ylim(ymax=0.3)
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
    # ax.set_ylim(ymax=0.3)
    plt.savefig(pngfile)
    plt.close()



    return





def plot_training_and_validation_accumulated_custom_one_figure(results,legends,outfile_base,colors,semilogy=False,fill_between=False,train_limits=None):
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
    mpl.rcParams['xtick.labelsize'] = 35
    mpl.rcParams['ytick.labelsize'] = 35
    # mpl.rcParams['xtick.labelsize'] = 25
    # mpl.rcParams['xtick.labelsize'] = 25
    # mpl.rcParams['font.family'] = 'Arial'


    njobs, nrep, nlosses, nepochs = results.shape
    x = np.arange(nepochs)
    M = np.sum(results,axis=3) > 0
    loss_types = ['loss','lossD']

    fig, ax = plt.subplots(figsize=(15,15))
    for i,loss in enumerate(loss_types):
        fig, ax = plt.subplots(figsize=(15, 15))
        for idx in range(2):
            for ii in range(njobs):
                if np.sum(M[ii, :, idx+i*2]) > 0:
                    y = results[ii, M[ii, :, idx+i*2], idx+i*2, :].mean(axis=0)
                    ystd = results[ii,M[ii,:,idx+i*2],idx+i*2,:].std(axis=0)
                    if semilogy:
                        # print(f"ii={ii},idx={idx},{colors[ii+njobs*idx]}")
                        if idx == 0:
                            h = ax.semilogy(x, y, '-', label=legends[ii*(idx+1)],color=colors[ii+njobs*idx])
                        else:
                            h = ax.semilogy(x, y, ':', label=legends[ii*(idx+1)],color=colors[ii+njobs*idx])
                    else:
                        h = ax.plot(x, y, '-', label=legends[ii*(idx+1)],color=colors[ii+njobs*idx])
                    if fill_between:
                        ax.fill_between(x, y - ystd, y+ystd, color=h[0].get_color(), alpha=0.2)

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # ax.set_yticks([20, 200, 500])
        y_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
        ax.yaxis.set_major_locator(y_major)
        y_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=100)
        ax.yaxis.set_minor_locator(y_minor)
        # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        if semilogy:
            pngfile = f"{outfile_base}_{loss}_semilogy.png"
        else:
            pngfile = f"{outfile_base}_{loss}.png"
        plt.savefig(pngfile, bbox_inches="tight", pad_inches=0)
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
    plt.clf()
    plt.close(fig)

    return

if __name__ == '__main__':
    file = ''