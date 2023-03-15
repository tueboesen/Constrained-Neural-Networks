import inspect
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

import torch
from e3nn import o3
from torch_cluster import radius_graph

from src.loss import loss_eq, loss_mim
from src.npendulum import get_coordinates_from_angle
from src.utils import Distogram, define_data_keys, atomic_masses
from src.vizualization import plot_pendulum_snapshot, plot_pendulum_snapshot_custom


def run_model(data_type,model, dataloader, train, max_samples, optimizer, loss_fnc, check_equivariance=False, max_radius=15, debug=False, epoch=None,output_folder=None,ignore_cons=False,nviz=5,regularization=0,viz_paper=False):
    """
    A wrapper function for the different data types currently supported for training/inference
    """
    if data_type == 'water':
        loss_r, loss_v, cv,cv_max ,MAEr, reg, reg2 = run_model_MD(model, dataloader, train, max_samples, optimizer, loss_fnc, check_equivariance=check_equivariance, max_radius=max_radius, debug=debug, epoch=epoch, output_folder=output_folder,ignore_con=ignore_cons,nviz=nviz,regularization=regularization,viz_paper=viz_paper)
        drmsd = 0
    elif data_type == 'pendulum' or data_type == 'n-pendulum' :
        loss_r, loss_v, cv,cv_max, cv_energy, cv_energy_max ,MAEr,MAEv, reg, reg2 = run_model_MD(model, dataloader, train, max_samples, optimizer, loss_fnc, check_equivariance=check_equivariance, max_radius=max_radius, debug=debug, epoch=epoch, output_folder=output_folder,ignore_con=ignore_cons,nviz=nviz,regularization=regularization,viz_paper=viz_paper)
        drmsd = 0
    else:
        raise NotImplementedError("The data_type={:}, you have selected is not implemented for {:}".format(data_type,inspect.currentframe().f_code.co_name))
    return loss_r, loss_v, drmsd, cv, cv_max,cv_energy,cv_energy_max,MAEr,MAEv,reg, reg2


def run_model_MD(model, dataloader, train, max_samples, optimizer, loss_type, check_equivariance=False, max_radius=15, debug=False,predict_pos_only=False, viz=True,epoch=None,output_folder=None,ignore_con=False,nviz=5,regularization=0,viz_paper=False):
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
    batch_size = dataloader.batch_size
    rscale = ds.rscale
    vscale = ds.vscale
    aloss_r = 0.0
    aloss_v = 0.0
    areg = 0.0
    areg2 = 0.0
    acv = 0.0
    acv_max = 0.0
    acv_energy = 0.0
    acv_energy_max = 0.0
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

        model_input = torch.cat([Rin_vec,Vin_vec],dim=-1)
        batch = torch.arange(Rin.shape[0]).repeat_interleave(Rin.shape[1]).to(device=Rin.device)

        m = m.view(nb,natoms,-1)
        weights = (1/m).repeat_interleave(ndim//m.shape[-1],dim=-1).repeat(1,1,model_input.shape[-1]//ndim)


        if ds.data_type == 'n-pendulum':
            n=5
            nb = (torch.max(batch)+1).item()
            a = torch.tensor([0])
            b = torch.arange(1,n-1).repeat_interleave(2)
            c = torch.tensor([n-1])
            I =torch.cat((a,b,c))

            bb1 = torch.arange(1,n)
            bb2 = torch.arange(n-1)
            J = torch.stack((bb1,bb2),dim=1).view(-1)

            shifts = torch.arange(nb).repeat_interleave(I.shape[0])*n

            II = I.repeat(nb)
            JJ = J.repeat(nb)

            edge_src = (JJ+shifts).to(device=Rin.device)
            edge_dst = (II+shifts).to(device=Rin.device)

            wstatic = torch.ones_like(edge_dst)

        else:
            edge_index = radius_graph(Rin_vec, max_radius, batch, max_num_neighbors=120)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            wstatic = None

        with P.cached():
            output, c,cv_max, reg, reg2 = model(model_input, batch, z_vec, edge_src, edge_dst,wstatic=wstatic,weight=weights)
        if output.isnan().any():
            raise ValueError("Output returned NaN")

        Rpred = output[:, 0:ndim]
        Vpred = output[:, ndim:]

        if ds.data_type == 'n-pendulum' and ndim == 1:
            x,y,vx,vy = get_coordinates_from_angle(Rpred.view(Rin.shape)[:,:,0].T, Vpred.view(Rin.shape)[:,:,0].T)
            x = x.T
            y = y.T
            vx = vx.T
            vy = vy.T
            Rpred = torch.cat((x[:, 1:, None], y[:, 1:, None]), dim=2).view(-1,2)
            Vpred = torch.cat((vx[:, 1:, None], vy[:, 1:, None]), dim=2).view(-1,2)

            x, y, vx, vy = get_coordinates_from_angle(Rin[:, :, 0].T, Vin[:, :, 0].T)
            x = x.T
            y = y.T
            vx = vx.T
            vy = vy.T
            Rin_vec = torch.cat((x[:, 1:, None], y[:, 1:, None]), dim=2).view(-1,2)
            Vin_vec = torch.cat((vx[:, 1:, None], vy[:, 1:, None]), dim=2).view(-1,2)

            x, y, vx, vy = get_coordinates_from_angle(Rout[:, :, 0].T, Vout[:, :, 0].T)
            x = x.T
            y = y.T
            vx = vx.T
            vy = vy.T
            Rout_vec = torch.cat((x[:, 1:, None], y[:, 1:, None]), dim=2).view(-1,2)
            Vout_vec = torch.cat((vx[:, 1:, None], vy[:, 1:, None]), dim=2).view(-1,2)


        lossE_r, lossE_ref_r, lossE_rel_r = loss_eq(Rpred, Rout_vec, Rin_vec)
        if lossE_rel_r > 100:
            print("ups")
        lossD_r, lossD_ref_r, lossD_rel_r = loss_mim(Rpred, Rout_vec, Rin_vec, edge_src, edge_dst)
        lossE_v, lossE_ref_v, lossE_rel_v = loss_eq(Vpred, Vout_vec, Vin_vec)
        lossD_v, lossD_ref_v, lossD_rel_v = loss_mim(Vpred, Vout_vec, Vin_vec, edge_src, edge_dst)
        if predict_pos_only:
            lossE_rel = lossE_rel_r
            lossD_rel = lossD_rel_r

        else:
            lossE_rel = (lossE_rel_r + lossE_rel_v)/2
            lossD_rel = (lossD_rel_r + lossD_rel_v)/2


        if loss_type.lower() == 'eq':
            loss_r, loss_v = lossE_rel_r, lossE_rel_v
            if predict_pos_only:
                loss_reg = reg2 * regularization
                loss = lossE_rel_r + loss_reg
                # loss = lossE_rel_r + torch.min(loss_reg,lossE_rel_r)
                # print(f"{lossE_rel_r:2.2f},{loss_reg:2.2f}")
            else:
                loss = lossE_rel + reg2 * regularization
        elif loss_type.lower() == 'mim':
            loss_r, loss_v = lossD_rel_r, lossD_rel_v
            if predict_pos_only:
                loss = lossD_rel_r + reg2 * regularization
            else:
                loss = lossD_rel + reg2 * regularization
        else:
            raise NotImplementedError("The loss function you have chosen has not been implemented.")

        # E_pred, Ekin_pred, Epot_pred, P_pred = energy_momentum(Vpred.view(Vout.shape) * vscale,Vpred.view(Vout.shape) * vscale, m) #TODO THIS DOESNT WORK JUST YET
        if ds.data_type == 'water':
            # rpred1 = (Rpred*rscale).view(-1,3,3)
            rdiff = ((Rpred - Rout_vec)*rscale).view(-1,3,3)
            MAEr = torch.mean(torch.norm(rdiff,dim=1))
        else:
            MAEr = torch.mean(torch.norm((Rpred - Rout_vec)*rscale,dim=1))
            MAEv = torch.mean(torch.norm((Vpred - Vout_vec)*vscale,dim=1))

        if check_equivariance:
            if ndim == 3:
                rot = o3.rand_matrix().to(device=x.device)
                Drot = model.irreps_in.D_from_matrix(rot)
                output_rot_after = output @ Drot
                output_rot = model(x @ Drot, batch, z_vec, edge_src, edge_dst)
                assert torch.allclose(output_rot,output_rot_after, rtol=1e-4, atol=1e-4)
            elif ndim == 2:
                theta = torch.rand(1)*torch.pi
                rot = torch.tensor([[torch.cos(theta), -torch.sin(theta)],[torch.sin(theta), torch.cos(theta)]])
                Rpred_rot_after = Rpred @ rot
                with P.cached():
                    model_input_rot = torch.cat([Rin_vec @ rot , Vin_vec @ rot], dim=-1)
                    output_rot, _,_,_,_ = model(model_input_rot, batch, z_vec, edge_src, edge_dst, wstatic=wstatic, weight=weights)
                    Rpred_rot = output_rot[:, 0:ndim]
                    assert torch.allclose(Rpred_rot, Rpred_rot_after, rtol=1e-4, atol=1e-4)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()



        aloss_r += loss_r.item()
        aloss_v += loss_v.item()

        if ds.data_type == 'n-pendulum' and natoms+1 == c.shape[1]:
            c_len = c[:,:-1]
            cabs_len = c_len.abs()
            c_energy = c[:,-1:]
            cabs_energy = c_energy.abs()
        else:
            cabs_len = c.abs()
            cabs_energy = torch.tensor(0.0)




        acv += cabs_len.mean().item()
        acv_max = max(cabs_len.max().item(),acv_max)

        acv_energy += cabs_energy.mean().item()
        acv_energy_max = max(cabs_energy.max().item(),acv_energy_max)
        areg += reg.item()
        areg2 += reg2.item()
        aMAEr += MAEr.item()
        aMAEv += MAEv.item()

        if ds.data_type == 'n-pendulum' and viz == True and i == 0:
            if ndim == 1:
                Rin_xy = Rin_vec.detach().cpu().view(nb,natoms,-1)
                Rout_xy = Rout_vec.detach().cpu().view(nb,natoms,-1)
                Rpred_xy = Rpred.detach().cpu().view(nb,natoms,-1)
                Vin_xy = Vin_vec.detach().cpu().view(nb,natoms,-1)
                Vout_xy = Vout_vec.detach().cpu().view(nb,natoms,-1)
                Vpred_xy = Vpred.detach().cpu().view(nb,natoms,-1)

            else:
                Rin_xy, Vin_xy, Rout_xy, Vout_xy, Rpred_xy, Vpred_xy = Rin.detach().cpu(), Vin.detach().cpu(), Rout.detach().cpu(), Vout.detach().cpu(), Rpred.detach().cpu().view(
                    Rin.shape), Vpred.detach().cpu().view(Rin.shape)

            for j in range(min(nviz,batch_size)):
                filename = f"{output_folder}/viz/{epoch}_{j}_{'train' if train==True else 'val'}_.png"
                plot_pendulum_snapshot(Rin_xy[j],Rout_xy[j],Vin_xy[j],Vout_xy[j],Rpred_xy[j],Vpred_xy[j],file=filename)

            if ignore_con and ndim > 1:
                with torch.no_grad():
                    output_no_con, reg, cv = model(model_input, batch, z_vec, edge_src, edge_dst, wstatic=wstatic,ignore_con=ignore_con)
                    Rpred_no_con = output_no_con[:, 0:ndim]
                    Vpred_no_con = output_no_con[:, ndim:]

                    Rpred_no_con_xy = Rpred_no_con.detach().cpu().view(Rin.shape)
                    Vpred_no_con_xy = Vpred_no_con.detach().cpu().view(Vin.shape)
                    for j in range(min(nviz,batch_size)):
                        filename = f"{output_folder}/viz/{epoch}_{j}_{'train' if train==True else 'val'}_ignore_con.png"
                        plot_pendulum_snapshot(Rin_xy[j],Rout_xy[j],Vin_xy[j],Vout_xy[j],Rpred_no_con_xy[j],Vpred_no_con_xy[j],file=filename)

        if ds.data_type == 'n-pendulum' and viz_paper == True:
            Rin_xy = Rin_vec.detach().cpu().view(nb, natoms, -1)
            Rout_xy = Rout_vec.detach().cpu().view(nb, natoms, -1)
            Rpred_xy = Rpred.detach().cpu().view(nb, natoms, -1)
            Vin_xy = Vin_vec.detach().cpu().view(nb, natoms, -1)
            Vout_xy = Vout_vec.detach().cpu().view(nb, natoms, -1)
            Vpred_xy = Vpred.detach().cpu().view(nb, natoms, -1)

            MAEr_xy = torch.norm((Rpred_xy - Rout_xy) * rscale, dim=-1)

            # lossE_r, lossE_ref_r, lossE_rel_r = loss_eq(Rpred_xy, Rout_xy, Rin_xy,reduce=False)
            # plot_pendulum_snapshots(Rin, Rout, Vin, Vout, Rpred=None, Vpred=None, file=file)
            for j in range(min(nviz, batch_size)):
                MAEj = MAEr_xy[j].mean()
                filename = f"{output_folder}/viz_paper/{j}_{MAEr_xy[j].mean():2.4f}.png"
                # fighandler = plot_pendulum_snapshot_custom(Rin_xy[j], Vin_xy[j], color='red')
                fighandler = plot_pendulum_snapshot_custom(Rout_xy[j],  color='green')
                fighandler = plot_pendulum_snapshot_custom(Rpred_xy[j], fighandler=fighandler, file=filename, color='blue')

    aloss_r /= (i + 1)
    aloss_v /= (i + 1)
    acv /= (i + 1)
    acv_energy /= (i + 1)
    aMAEr /= (i + 1)
    areg /= (i + 1)
    areg2 /= (i + 1)
    aMAEv /= (i + 1)

    return aloss_r, aloss_v, acv, acv_max, acv_energy, acv_energy_max, aMAEr,aMAEv, areg, areg2#, aMAEv