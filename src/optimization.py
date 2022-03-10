import inspect
import torch.nn.functional as F

import torch
from e3nn import o3
from torch_cluster import radius_graph

from src.loss import loss_eq, loss_mim


def run_model(data_type,model, dataloader, train, max_samples, optimizer, loss_fnc, batch_size=1, check_equivariance=False, max_radius=15, debug=False):
    """
    A wrapper function for the different data types currently supported for training/inference
    """
    if data_type == 'water':
        loss, lossD = run_model_MD(model, dataloader, train, max_samples, optimizer, loss_fnc, batch_size=batch_size, check_equivariance=check_equivariance, max_radius=max_radius, debug=debug)
    elif data_type == 'protein':
        loss, lossD = run_model_protein(model,dataloader,train,max_samples,optimizer, loss_fnc, batch_size=1)
    else:
        raise NotImplementedError("The data_type={:}, you have selected is not implemented for {:}".format(data_type,inspect.currentframe().f_code.co_name))
    return loss, lossD


def run_model_MD(model, dataloader, train, max_samples, optimizer, loss_type, batch_size=1, check_equivariance=False, max_radius=15, debug=False):
    """
    A function designed to optimize or test a model on molecular dynamics data. Note this function will only run a maximum of one full sweep through the dataloader (1 epoch)

    model:          a handler to the neural network used
    dataloader:     a handler to the dataloader used
    train:          a boolean determining whether to run in training or evaluation mode
    max_samples:    the maximum number of samples to run before exiting (typically used when validating to only draw a smaller subset of a large dataset)
    optimizer:      the optimizing function used to train the model
    loss_type:      a string that determines which loss to use. ('mim','eq')
    batch_size:     the batch_size to use during the run
    check_equivariance:     Checks whether the network is equivariant, should only be used for debugging.
    max_radius:     The maximum radius used when building the edges in the graph between the particles
    debug:          Checks for NaNs and inf in the network while running.
    """
    ds = dataloader.dataset
    rscale = ds.rscale
    vscale = ds.vscale
    aloss = 0.0
    alossD = 0.0
    aMAEr = 0.0
    aMAEv = 0.0
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    for i, (Rin, Rout, z, Vin, Vout, Fin, Fout, KEin, KEout, PEin, PEout, m) in enumerate(dataloader):
        nb, natoms, ndim = Rin.shape
        optimizer.zero_grad()

        Rin_vec = Rin.reshape(-1,Rin.shape[-1])
        Rout_vec = Rout.reshape(-1,Rout.shape[-1])
        Vin_vec = Vin.reshape(-1,Vin.shape[-1])
        Vout_vec = Vout.reshape(-1,Vout.shape[-1])
        z_vec = z.reshape(-1,z.shape[-1])

        x = torch.cat([Rin_vec,Vin_vec],dim=-1)
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=120)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        output, reg = model(x, batch, z_vec, edge_src, edge_dst)
        if output.isnan().any():
            raise ValueError("Output returned NaN")

        Rpred = output[:, 0:ndim]
        Vpred = output[:, ndim:]

        loss_r, loss_ref_r, loss_rel_r = loss_eq(Rpred, Rout_vec, Rin_vec)
        loss_v, loss_ref_v, loss_rel_v = loss_eq(Vpred, Vout_vec, Vin_vec)
        loss_rel = (loss_rel_r + loss_rel_v)/2

        lossD_r, lossD_ref_r, lossD_rel_r = loss_mim(Rpred, Rout_vec, Rin_vec, edge_src, edge_dst)
        lossD_v, lossD_ref_v, lossD_rel_v = loss_mim(Vpred, Vout_vec, Vin_vec, edge_src, edge_dst)
        lossD_rel = (lossD_rel_r + lossD_rel_v)/2

        # E_pred, Ekin_pred, Epot_pred, P_pred = energy_momentum(Vpred.view(Vout.shape) * vscale,Vpred.view(Vout.shape) * vscale, m) #TODO THIS DOESNT WORK JUST YET

        MAEr = torch.mean(torch.abs(Rpred - Rout_vec)*rscale).detach()
        MAEv = torch.mean(torch.abs(Vpred - Vout_vec)*vscale).detach()

        if check_equivariance:
            rot = o3.rand_matrix().to(device=x.device)
            Drot = model.irreps_in.D_from_matrix(rot)
            output_rot_after = output @ Drot
            output_rot = model(x @ Drot, batch, z_vec, edge_src, edge_dst)
            assert torch.allclose(output_rot,output_rot_after, rtol=1e-4, atol=1e-4)
            print("network is equivariant")
        if train:
            if loss_type.lower() == 'eq':
                loss = loss_rel + reg
            elif loss_type.lower() == 'mim':
                loss = lossD_rel + reg
            else:
                raise NotImplementedError("The loss function you have chosen has not been implemented.")

            loss.backward()
            if debug:
                weights = optimizer.param_groups[0]['params']
                weights_flat = [torch.flatten(weight) for weight in weights]
                weights_1d = torch.cat(weights_flat)
                assert not torch.isnan(weights_1d).any()
                assert not torch.isinf(weights_1d).any()
                print(f"{weights_1d.max()}, {weights_1d.min()}")

                grad_flat = [torch.flatten(weight.grad) for weight in weights]
                grad_1d = torch.cat(grad_flat)
                assert not torch.isnan(grad_1d).any()
                assert not torch.isinf(grad_1d).any()
                print(f"{grad_1d.max()}, {grad_1d.min()}")
            optimizer.step()

        aloss += loss_rel.detach()
        alossD += lossD_rel.detach()
        aMAEr += MAEr
        aMAEv += MAEv
        if (i + 1) * batch_size >= max_samples:
            break

    aloss /= (i + 1)
    alossD /= (i + 1)
    aMAEr /= (i + 1)
    aMAEv /= (i + 1)

    return aloss, alossD#, aMAEr, aMAEv


def run_model_protein(model,dataloader,train,max_samples,optimizer, loss_type, batch_size=1, debug=False):
    """
    A function designed to optimize or test a model on protein predictions data. Note this function will only run a maximum of one full sweep through the dataloader (1 epoch)

    model:          a handler to the neural network used
    dataloader:     a handler to the dataloader used
    train:          a boolean determining whether to run in training or evaluation mode
    max_samples:    the maximum number of samples to run before exiting (typically used when validating to only draw a smaller subset of a large dataset)
    optimizer:      the optimizing function used to train the model
    loss_type:      a string that determines which loss to use. ('mim','eq')
    batch_size:     the batch_size to use during the run. NOTE THAT THE CURRENT CODE ONLY SUPPORTS BATCH_SIZE OF 1 FOR PROTEINS.
    """
    aloss = 0.0
    alossD = 0.0
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
    for i, (seq,batch, coords, M, edge_index,edge_index_all) in enumerate(dataloader):
        if torch.sum(M) < 5 or len(edge_index_all[0]) == 0:
            continue # We skip proteins where there are 5 or less known amino acids in, and where there are no edge connections, this should never really be a problem but in unsanitized datasets it might be a problem
        nb = len(torch.unique(batch))
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_src_all = edge_index_all[0]
        edge_dst_all = edge_index_all[1]

        optimizer.zero_grad()

        # We need to define a best guess for the amino acid coordinates
        coords_init = 0.1*torch.ones_like(coords)
        coords_init[::2,::3] += 0.1
        coords_init[:,3:] += 0.5
        coords_init[:,6:] += 0.5
        coords_init = torch.cumsum(coords_init,dim=0)

        coords_pred, reg = model(x=coords_init,batch=batch,node_attr=seq, edge_src=edge_src, edge_dst=edge_dst)

        loss, loss_ref, loss_rel = loss_eq(coords_pred, coords, coords*0) #Note that we don't use the inital guess here, but rather 0.
        lossD, lossD_ref, lossD_rel = loss_mim(coords_pred, coords, coords*0, edge_src_all, edge_dst_all) #Note that we don't use the inital guess here, but rather 0.

        if loss_type.lower() == 'eq':
            loss = loss_rel + reg
        elif loss_type.lower() == 'mim':
            loss = lossD_rel + reg
        else:
            raise NotImplementedError("The loss_fnc is not implemented.")
        if train:
            loss.backward()
            optimizer.step()
            if debug:
                weights = optimizer.param_groups[0]['params']
                weights_flat = [torch.flatten(weight) for weight in weights]
                weights_1d = torch.cat(weights_flat)
                assert not torch.isnan(weights_1d).any()
                assert not torch.isinf(weights_1d).any()

                grad_flat = [torch.flatten(weight.grad) for weight in weights]
                grad_1d = torch.cat(grad_flat)
                assert not torch.isnan(grad_1d).any()
                assert not torch.isinf(grad_1d).any()

        aloss += loss_rel.detach()
        alossD += lossD_rel.detach()

        if (i + 1) * batch_size >= max_samples:
            break

    aloss /= (i + 1)
    alossD /= (i + 1)
    return aloss, alossD



