import multibodypendulum as mbp
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

    dataset_train = vector_field(images[:n_train])
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batchsize_train)
    dataloaders['train'] = dataloader_train

    if use_val:
        dataset_val = vector_field(images[n_train:n_train+n_val])
        dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batchsize_val, drop_last=False)
        dataloaders['val'] = dataloader_val

    if use_test:
        dataset_test = vector_field(images[n_train+n_val:n_train+n_val+n_test])
        dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=batchsize_val, drop_last=False)
        dataloaders['test'] = dataloader_test

    return dataloaders



class vector_field(Dataset):
    def __init__(self, UVdata):
        self.UV = UVdata

    def __getitem__(self, idx):
        UV = self.UV[idx]
        return UV

    def __len__(self):
        return self.UV.shape[0]

