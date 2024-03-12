# import multibodypendulum as mbp
import os
from os.path import exists

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataloader_utils import data_split, generate_dataloaders, attach_edge_generator


def load_imagedenoising_data(file, use_val, use_test, metafile, n_train, n_val, n_test, batchsize_train, batchsize_val, nskip, device, data_id):
    """
    Loads a imagedenoising dataset into standard pytorch dataloaders.
    """
    dataloaders = {}

    with np.load(file) as data:
        images = data['images']
        div = data['divergence']
    images = torch.from_numpy(images)
    images = images.to(device=device,dtype=torch.get_default_dtype())

    if exists(metafile):
        with np.load(metafile) as mf:
            ndata = mf['ndata']
            assert images.shape[0] == ndata, "The number of data points in the dataset does not match the number in the metadata."
            idx_train = mf['idx_train']
            idx_val = mf['idx_val']
            idx_test = mf['idx_test']
    else:
        ndata = images.shape[0]
        ndata_rand = 0 + np.arange(ndata)
        np.random.shuffle(ndata_rand)
        idx_train = ndata_rand[:n_train]
        idx_val = ndata_rand[n_train:n_train + n_val]
        idx_test = ndata_rand[n_train + n_val:n_train + n_val + n_test]
        os.makedirs(os.path.dirname(metafile), exist_ok=True)
        np.savez(metafile, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, ndata=ndata)

    dataset_train = vector_field(images[idx_train])
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batchsize_train)
    dataloaders['train'] = dataloader_train

    if use_val:
        dataset_val = vector_field(images[idx_val])
        dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batchsize_val, drop_last=False)
        dataloaders['val'] = dataloader_val

    if use_test:
        dataset_test = vector_field(images[idx_test])
        dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=batchsize_val, drop_last=False)
        dataloaders['test'] = dataloader_test
    return dataloaders



class vector_field(Dataset):
    def __init__(self, UVdata, noise_level=0.1):
        self.UV = UVdata
        self.noise_level = noise_level

    def __getitem__(self, idx):
        UV = self.UV[idx]
        UV_noise = UV + self.noise_level * torch.randn_like(UV)
        return UV, UV_noise

    def __len__(self):
        return self.UV.shape[0]

