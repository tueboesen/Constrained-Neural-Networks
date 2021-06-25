import numpy as np
import torch

from src.utils import convert_snapshots_to_future_state_dataset

folder = './../../../data/covid/'
filein = 'covid_backbone.npz'
nskip = 0
ntrain = 10000
nval = 1000

data = np.load(folder+filein, allow_pickle=True)
Ra = torch.from_numpy(data['RCA']).to(dtype=torch.get_default_dtype())
Rb = torch.from_numpy(data['RCB']).to(dtype=torch.get_default_dtype())
Rn = torch.from_numpy(data['RN']).to(dtype=torch.get_default_dtype())
z = torch.from_numpy(data['aa_num']).to(dtype=torch.int64)
fragids = torch.from_numpy(data['fragid']).to(dtype=torch.get_default_dtype())
R_org = torch.cat([Ra, Rb, Rn], dim=2)
R = R_org[1:]
V = R_org[1:] - R_org[:-1]
nz = len(z.unique())

fragid_unique = torch.unique(fragids)
d_nn = 3.8 * torch.ones(nz, nz)
count_nn = torch.ones((nz, nz), dtype=torch.int64)
d_ab = torch.zeros(nz)
count_ab = torch.zeros(nz, dtype=torch.int64)
d_an = torch.zeros(nz)
count_an = torch.zeros(nz, dtype=torch.int64)

for fragid_i in fragid_unique:
    idx = fragid_i == fragids
    Ri = R[:, idx, :]
    dRi = Ri[:, 1:, :] - Ri[:, :-1, :]
    dRa = dRi[:, :, :3]
    for i in range(dRa.shape[1]):
        d_nn[z[i], z[i + 1]] += torch.sum(torch.sqrt(torch.sum(dRa[:, i, :] ** 2, dim=-1)))
        count_nn[z[i], z[i + 1]] += dRa.shape[0]
    dRab = Ri[:, :, :3] - Ri[:, :, 3:6]
    dRan = Ri[:, :, :3] - Ri[:, :, 6:9]
    for i in range(Ri.shape[1]):
        d_ab[z[i]] += torch.sum(torch.sqrt(torch.sum(dRab[:, i, :] ** 2, dim=-1)))
        count_ab[z[i]] += dRab.shape[0]
        d_an[z[i]] += torch.sum(torch.sqrt(torch.sum(dRan[:, i, :] ** 2, dim=-1)))
        count_an[z[i]] += dRan.shape[0]

    distRa = torch.norm(dRa, dim=2)
    distRab = torch.norm(dRab, dim=2)
    distRan = torch.norm(dRan, dim=2)
    print(
        f"max={distRa.max():3.2f}, min={distRa.min():3.2f}, mean={distRa.mean():3.2f}. AlphaBeta_Dist(mean) = {distRab.mean():3.2f}, AlphaBeta_Dist(min) = {distRab.min():3.2f}, AlphaBeta_Dist(max) = {distRab.max():3.2f}, AlphaN_Dist(mean) = {distRan.mean():3.2f},AlphaN_Dist(min) = {distRan.min():3.2f} AlphaN_Dist(max) = {distRan.max():3.2f}")

dist_nn = d_nn / count_nn
dist_ab = d_ab / count_ab
dist_an = d_an / count_an

# dist_nnz = dist_nn[z[:-1],z[1:]]
dist_abz = dist_ab[z]
dist_anz = dist_an[z]

np.save("{:}dist_abz".format(folder),dist_abz)
np.save("{:}dist_anz".format(folder),dist_anz)

Rin, Rout = convert_snapshots_to_future_state_dataset(nskip, R)
Vin, Vout = convert_snapshots_to_future_state_dataset(nskip, V)

# p = torch.sum(V * masses[None,:,None],dim=1)

ndata = np.min([Rout.shape[0]])
natoms = z.shape[0]

print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

ndata_rand = 0 + np.arange(ndata)
np.random.shuffle(ndata_rand)
train_idx = ndata_rand[:ntrain]
val_idx = ndata_rand[ntrain:ntrain + nval]
test_idx = ndata_rand[ntrain + nval:]

Rin_train = Rin[train_idx]
Rout_train = Rout[train_idx]
Vin_train = Vin[train_idx]
Vout_train = Vout[train_idx]
dataset_train = DatasetFutureState(Rin_train, Rout_train, z, Vin_train, Vout_train, device=device)
dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)
