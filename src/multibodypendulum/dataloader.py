# import multibodypendulum as mbp
import numpy as np
import torch

from src.dataloader_utils import data_split, generate_dataloaders, attach_edge_generator
from src.utils import convert_snapshots_to_future_state_dataset
from src.vizualization import plot_pendulum_snapshot


def load_multibodypendulum_data(file, use_val, use_test, metafile, n_train, n_val, n_test, batchsize_train, batchsize_val, nskip, device, data_id):
    """
    Creates a multibody pendulum dataset and loads it into standard pytorch dataloaders.
    #TODO we should save this as a feature artifact that can be loaded and applied directly to the raw data
    """

    features = feature_transform_multibodypendulum(file, nskip, device)
    f_train, f_val, f_test = data_split(features, metafile, n_train, n_val, n_test)
    dataloaders = generate_dataloaders(f_train, f_val, f_test, batchsize_train, batchsize_val, use_val, use_test)
    dataloaders = attach_edge_generator(dataloaders, multibodypendulum_edges)
    return dataloaders


def feature_transform_multibodypendulum(file, nskip, device):
    """
    TODO save this function as an artifact with the models for reproduceability
    """
    with np.load(file) as data:
        theta = data['theta']
        dtheta = data['dtheta']
    theta = torch.from_numpy(theta)
    dtheta = torch.from_numpy(dtheta)
    x, y, vx, vy = mbp.MultiBodyPendulum.get_coordinates_from_angles(theta, dtheta)

    R = torch.cat((x.T[:, None, :, None], y.T[:, None, :, None]), dim=-1)
    V = torch.cat((vx.T[:, None, :, None], vy.T[:, None, :, None]), dim=-1)
    Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
    Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)
    particle_type = (torch.arange(Rin.shape[2]) + 1)[None, :].repeat(Rin.shape[0], 1)
    # particle_type = torch.ones_like(Rin).repeat(1,1,1,2) # we repeat it twice for R and V, which constitute our vector at the end
    particle_mass = torch.ones_like(Rin).repeat(1, 1, 1, 2)
    # particle_type = torch.ones((Rin.shape[0],Rin.shape[1],Rin.shape[2]))
    # particle_mass = torch.ones((Rin.shape[0],Rin.shape[1],Rin.shape[2]))

    features = {}
    features['Rin'] = Rin.to(device)
    features['Rout'] = Rout.to(device)
    features['Vin'] = Vin.to(device)
    features['Vout'] = Vout.to(device)
    features['particle_type'] = particle_type.to(device)
    features['particle_mass'] = particle_mass.to(device)
    return features


def multibodypendulum_edges(batch, x, max_radius, npenduls=5):
    """
    For a multibody pendulum we replace the default edge connector with a custom edge generator which connects each pendulum (node) with its neighboring pendulums rather than all pendulums within a certain radius.
    """
    # Rin_vec = x[...,:x.shape[-1]//2]
    nb = (torch.max(batch) + 1).item()
    a = torch.tensor([0])
    b = torch.arange(1, npenduls - 1).repeat_interleave(2)
    c = torch.tensor([npenduls - 1])
    I = torch.cat((a, b, c))

    bb1 = torch.arange(1, npenduls)
    bb2 = torch.arange(npenduls - 1)
    J = torch.stack((bb1, bb2), dim=1).view(-1)

    shifts = torch.arange(nb).repeat_interleave(I.shape[0]) * npenduls

    II = I.repeat(nb)
    JJ = J.repeat(nb)

    edge_src = (JJ + shifts).to(device=batch.device)
    edge_dst = (II + shifts).to(device=batch.device)

    wstatic = torch.ones_like(edge_dst)
    return edge_src, edge_dst, wstatic


if __name__ == "__main__":
    file = './../../data/multibodypendulum/multibodypendulum.npz'
    nskip = 100
    device = 'cpu'
    features = feature_transform_multibodypendulum(file, nskip, device)
    Rins = features['Rin']
    Routs = features['Rout']
    Vins = features['Vin']
    Vouts = features['Vout']
    n = Rins.shape[0]
    indices = list(range(62345, n, nskip))
    for idx in indices:
        fileout = f'./../../../results/pendulums/idx_{idx}_nskip_{nskip}.png'
        Rin = Rins[idx].squeeze()
        Rout = Routs[idx].squeeze()
        Vin = Vins[idx].squeeze()
        Vout = Vouts[idx].squeeze()
        Rpred = Routs[idx + nskip].squeeze()
        Vpred = Vouts[idx + nskip].squeeze()
        plot_pendulum_snapshot(Rin, Vin, Rout=None, Vout=None, Rpred=None, Vpred=None, file=fileout)
