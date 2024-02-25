from os import listdir
from os.path import isfile, join
import os
from collections import OrderedDict
from collections import namedtuple
import matplotlib as mpl
import mlflow

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from src.constraints import generate_constraints
from src.dataloader import load_data_wrapper
from src.imagedenoising.create_imagedenoising_data import div_cc
from src.imagedenoising.dataloader import load_imagedenoising_data
from src.network_resnet import neural_network_resnet
from src.networks import generate_neural_network
from src.optimization import generate_constraints_minimizer
import torch.nn.functional as F


class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

def read_meta_yaml(file):
    conf = OmegaConf.load(file)
    return conf

def read_metric_file(file):
    data = pd.read_csv(file,header=None,sep=' ')
    metric = data.iloc[:,1]
    epochs = data.iloc[:,2]
    return metric,epochs

def get_metric(name,metric_folder):
    file = os.path.join(metric_folder, name)
    metric, epochs = read_metric_file(file)
    metric.name = name
    return metric, epochs

def read_param(file):
    with open(file, 'r') as f:
        param = f.readline()
    return param

def read_all_params(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    params = {}
    for file in files:
        with open(join(path,file), 'r') as f:
            param = f.readline()
            try:
                param = int(param)
            except:
                try:
                    param = float(param)
                except:
                    pass

        names = file.split('.')
        if names[0] not in params:
            params[names[0]] = {}
        params[names[0]][names[1]] = param
    # params_obj = obj(params)

    return params

def save_image(name,image):
    fig = plt.figure(frameon=False)
    w = 2
    fig.set_size_inches(w, w)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')
    fig.savefig(name)


def plot_mlflow_artifact(mlflow_paths,dataloaders,metric_name, seed):
    # metric_name = 'cv_mean'
    # y_label = 'Constraint violation (m)'

    names = []
    metrics = {}
    counter = 0
    images, images_noisy = next(iter(dataloaders['test']))
    dataloaders['test'].dataset.noise_level *= 10
    images2, images_noisy10x = next(iter(dataloaders['test']))
    dataloaders['test'].dataset.noise_level /= 100
    images3, images_noisy01x = next(iter(dataloaders['test']))
    images_pred = []
    images_pred10x = []
    images_pred01x = []
    losses0 = []
    losses0m = []
    losses1 = []
    losses1m = []

    for path in mlflow_paths:
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        for folder in subfolders:
            meta_file = os.path.join(folder,'meta.yaml')
            conf = read_meta_yaml(meta_file)
            if conf.status != 3 or conf.lifecycle_stage != 'active': # Run didn't finish
                continue
            if seed != int(read_param(f"{folder}/params/run.seed")):
                continue
            name = conf.run_name
            if name in names:
                continue
            names.append(name)

            path_params = os.path.join(folder,'params')
            params = read_all_params(path_params)

            if params['run']['precision'] == 'torch.float64':
                torch.set_default_dtype(torch.float64)


            con_fnc = generate_constraints(params['constraint'])
            min_con_fnc = generate_constraints_minimizer(params['minimization'], con_fnc)
            model = generate_neural_network(params['model'], con_fnc=min_con_fnc)

            metric_folder = f"{folder}/metrics"
            metric_train, epochs = get_metric(f"train_{metric_name}",metric_folder)
            metric_val, epochs = get_metric(f"val_{metric_name}",metric_folder)
            state_dict_path = f"{folder}/artifacts/models"
            state_dict = mlflow.pytorch.load_state_dict(state_dict_path)

            device = params['run']['device']

            model.load_state_dict(state_dict)
            model.train(False)
            model.to(device=device)

            images_noisy = images_noisy.to(device=device, dtype=torch.get_default_dtype())
            images_noisy01x = images_noisy01x.to(device=device, dtype=torch.get_default_dtype())
            images_noisy10x = images_noisy10x.to(device=device, dtype=torch.get_default_dtype())

            if images_noisy.ndim == 3:
                images_noisy = images_noisy[None]
            with torch.no_grad():
                x_pred, cv_mean, cv_max, reg = model(images_noisy)
                x_pred10x, cv_mean, cv_max, reg = model(images_noisy10x)
                x_pred01x, cv_mean, cv_max, reg = model(images_noisy01x)
            x_pred = x_pred.cpu()
            images_pred.append(x_pred.cpu())
            images_pred10x.append(x_pred10x.cpu())
            images_pred01x.append(x_pred01x.cpu())
            loss0 = F.mse_loss(x_pred[:,0],images[:,0],reduction='none')
            loss1 = F.mse_loss(x_pred[:,1],images[:,1],reduction='none')
            loss0 = loss0.mean(dim=(1,2))
            loss1 = loss1.mean(dim=(1,2))
            losses0.append(loss0)
            losses1.append(loss1)
            losses0m.append(loss0.mean().item())
            losses1m.append(loss1.mean().item())

    images_pred = torch.stack(images_pred)
    images_pred10x = torch.stack(images_pred10x)
    images_pred01x = torch.stack(images_pred01x)

    idx = 0

    images_pred = images_pred[:,idx]
    images_pred10x = images_pred10x[:,idx]
    images_pred01x = images_pred01x[:,idx]
    images = images[idx]
    images_noisy = images_noisy[idx]
    images_noisy10x = images_noisy10x[idx]
    images_noisy01x = images_noisy01x[idx]


    # images_pred = images_pred[:,0]
    div = div_cc(images_pred)
    # shape = images_pred.shape
    # div = div_cc(images_pred.view(-1,*images.shape[1:])).view(shape[0],shape[1],shape[3]-1,shape[4]-1)
    # div = div_cc(images_pred.view(-1,*images.shape[1:])).view(shape[0],shape[1],shape[3]-1,shape[4]-1)
    images = images.cpu()
    images_noisy = images_noisy.cpu()
    images_noisy10x = images_noisy10x.cpu()
    images_noisy01x = images_noisy01x.cpu()
    dimage = images_pred - images
    ndiv = div / torch.max(torch.abs(div))
    ndimage = dimage / torch.max(torch.abs(dimage))
    # losses0 = torch.tensor(losses0)
    # losses1 = torch.tensor(losses1)


    # fig, axs = plt.subplots(8,7, figsize=(15, 15))

    save_image('im0.png', images[0])
    save_image('im1.png', images[1])
    save_image('im0n.png', images_noisy[0])
    save_image('im1n.png', images_noisy[1])
    save_image('im0n10x.png', images_noisy10x[0])
    save_image('im1n10x.png', images_noisy10x[1])
    save_image('im0n01x.png', images_noisy01x[0])
    save_image('im1n01x.png', images_noisy01x[1])


    for i in range(len(names)):
        # pim0 = axs[0,i+2].imshow(images_pred[i,0])
        # pim1 = axs[1,i+2].imshow(images_pred[i,1])
        # axs[2, i + 2].imshow(dimage[i, 0])
        # axs[3, i + 2].imshow(dimage[i, 1])
        # axs[4, i + 2].imshow(div[i,0])
        # axs[5, i + 2].imshow(ndimage[i, 0], vmin=-1, vmax=1)
        # axs[6, i + 2].imshow(ndimage[i, 1], vmin=-1, vmax=1)
        # axs[7, i + 2].imshow(ndiv[i,0], vmin=-1, vmax=1)

        name = f"0_{names[i]}.png"
        save_image(name, images_pred[i,0])

        name = f"1_{names[i]}.png"
        save_image(name, images_pred[i,1])

        name = f"0_{names[i]}_10x.png"
        save_image(name, images_pred10x[i,0])

        name = f"1_{names[i]}_10x.png"
        save_image(name, images_pred10x[i,1])

        name = f"0_{names[i]}_01x.png"
        save_image(name, images_pred01x[i,0])

        name = f"1_{names[i]}_01x.png"
        save_image(name, images_pred01x[i,1])



    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # axs[0,0].title.set_text("Image")
    # axs[0,1].title.set_text("Noisy Image")
    # axs[0,2].title.set_text("Regularization")
    # axs[0,3].title.set_text("No constraint")
    # axs[0,4].title.set_text("Penalty")
    # axs[0,5].title.set_text("End constraint")
    # axs[0,6].title.set_text("Smooth constraint")
    #
    # axs[0,0].set_ylabel("Image 1")
    # axs[1,0].set_ylabel("Image 2")
    # axs[2,0].set_ylabel("dImage 1")
    # axs[3,0].set_ylabel("dImage 2")
    # axs[4,0].set_ylabel("Constraint violation")
    # axs[5,0].set_ylabel("norm dImage 1")
    # axs[6,0].set_ylabel("norm dImage 2")
    # axs[7,0].set_ylabel("norm Constraint violation")


    # plt.pause(1)
    #
    #
    # prefix = f"{name}"
    # pngfile = f"{prefix}_{metric_name}.png"
    # pngfile = f"comparison2.png"
    # fig.savefig(pngfile)
    # plt.close()



    return

if __name__ == "__main__":

    legend_dict = OrderedDict({
        'imagedenoising_-1__0_0': 'No Constraints',
        'imagedenoising_-1__0_0.1': 'Auxiliary loss $\eta=0.1$',
        'imagedenoising_-1__0_1': 'Auxiliary loss $\eta=1$',
        'imagedenoising_-1__0_10': 'Auxiliary loss $\eta=10$',
        'imagedenoising_-1__0.1_0': 'Penalty $\gamma=0.1$',
        'imagedenoising_-1__1_0': 'Penalty $\gamma=1$',
        'imagedenoising_-1__10_0': 'Penalty $\gamma=10$',
        'imagedenoising_-1_low_0_1': 'End Constraints $\eta=1$',
        'imagedenoising_-1_low_0_0.1': 'End Constraints $\eta=0.1$',
        'imagedenoising_-1_high_0_1': 'Smooth Constraints $\eta=1$',
        'imagedenoising_-1_high_0_0.1': 'Smooth Constraints $\eta=0.1$',
        'imagedenoising_-1_high_0_0.01': 'Smooth Constraints $\eta=0.01$',
    })
    mlflow_path = ['/home/tue/PycharmProjects/results/mlflow/818020475320564793'] #614286386201480341
    # mlflow_path = ['/home/tue/PycharmProjects/results/mlflow/453210295369231837'] #614286386201480341
    prefix = 'imagedenoising'
    metric_name = 'cv_mean'
    y_label = 'Constraint violation (m)'
    images_path = './../../data/imagedenoising/images.npz'
    data = np.load(images_path)
    images = torch.from_numpy(data['images'])

    use_val = True
    use_test = True
    seed = 1234
    n_train = 100
    n_val = 100
    n_test = 100
    batchsize_train = 10
    batchsize_val = 100
    batchsize_test = 100
    nskip = 0.1
    device = 'cpu'
    data_id = None
    metafile = f'./../../data/imagedenoising/metadata/split_{seed}_{n_train}_{n_val}_{n_test}_{nskip}.npz'
    dataloaders = load_imagedenoising_data(images_path, use_val, use_test, metafile, n_train, n_val, n_test, batchsize_train, batchsize_val, nskip, device, data_id)



    plot_mlflow_artifact(mlflow_path,dataloaders,metric_name, seed)


