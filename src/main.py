import os
import time
from datetime import datetime

import torch

from src import log
from src.constraints import load_constraints, load_constraint_parameters
from src.dataloader import load_data
from src.log import log_all_parameters, close_logger
from src.loss import find_relevant_loss
from src.network_e3 import neural_network_equivariant
from src.network_mim import neural_network_mimetic
from src.optimization import run_model
from src.project_uplift import ProjectUpliftEQ, ProjectUplift
from src.utils import fix_seed, update_results_and_save_to_csv, run_model_MD_propagation_simulation
from src.vizualization import plot_training_and_validation


def main(c,dataloader_train=None,dataloader_val=None,dataloader_test=None,dataloader_endstep=None):
    """
    The main function which should be called with a fitting configuration dictionary.
    c is the input configuration dictionary, which dictates the run.
    """
    cn = c['network']
    if c['use_double']:
        torch.set_default_dtype(torch.float64)
    fix_seed(c['seed'])  # Set a seed, so we make reproducible results.
    if 'result_dir' not in c:
        c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
            root='results',
            runner_name=c['basefolder'],
            date=datetime.now(),
        )
    model_name = "{}/{}.pt".format(c['result_dir'], 'model')
    model_name_best = "{}/{}.pt".format(c['result_dir'], 'model_best')
    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner', logfile_loc, c['debug'])
    log_all_parameters(LOG, c)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load training data
    if dataloader_train is None:
        dataloader_train, dataloader_val, dataloader_test, dataloader_endstep = load_data(c['data'], c['data_type'], device, c['nskip'], c['n_train'], c['n_val'], c['use_val'], c['use_test'], c['batch_size'], use_endstep=c['perform_endstep_MD_propagation'])
    ds = dataloader_train.dataset

    if c['network_type'].lower() == 'eq':
        PU = ProjectUpliftEQ(cn['irreps_inout'], cn['irreps_hidden'])
    elif c['network_type'].lower() == 'mim':
        node_dim_in = cn['node_dim_in'] if ds.pos_only else cn['node_dim_in']*2
        PU = ProjectUplift(node_dim_in, cn['node_dim_latent'])

    cv = load_constraint_parameters(c['con'], c['con_type'], c['data_type'], con_data=c['con_data'])

    #PU, masses=ds.m, R=ds.Rin, V=ds.Vin, z=ds.z, rscale=ds.rscale, vscale=ds.vscale, energy_predictor=c['PE_predictor']
    con_fnc = load_constraints(c['con'], c['con_type'], project_fnc=PU.project, uplift_fnc=PU.uplift, debug=c['debug'], con_variables=cv,rscale=ds.rscale,vscale=ds.vscale,pos_only=ds.pos_only,regularizationparameter=c['regularizationparameter'])

    if c['network_type'].lower() == 'eq':
        model = neural_network_equivariant(irreps_inout=cn['irreps_inout'], irreps_hidden=cn['irreps_hidden'], layers=cn['layers'],
                                    max_radius=cn['max_radius'],
                                    number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                                    num_nodes=ds.Rin.shape[1], embed_dim=cn['embed_dim'], max_atom_types=cn['max_atom_types'], con_fnc=con_fnc, con_type=c['con_type'], PU=PU, particles_pr_node=ds.particles_pr_node)
    elif c['network_type'] == 'mim':
        model = neural_network_mimetic(cn['node_dim_latent'], cn['nlayers'], PU=PU, con_fnc=con_fnc, con_type=c['con_type'],dim=c["data_dim"])
    else:
        raise NotImplementedError("Network type is not implemented")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    LOG.info('Number of parameters {:}'.format(total_params))

    lr = c['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossBest = 1e6
    epochs_since_best = 0
    results=None
    epoch = 0
    csv_file = f"{c['result_dir']}/training.csv"

    t0 = time.time()
    # print(f"{torch.cuda.memory_allocated() / 1024 / 1024:2.2f}MB")
    # print(f"{torch.cuda.max_memory_allocated() / 1024 / 1024:2.2f}MB")
    while epoch < c['epochs']:
        t1 = time.time()
        loss_r_t, loss_v_t = run_model(c['data_type'], model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size'], max_radius=cn['max_radius'], debug=c['debug'])
        t2 = time.time()
        if c['use_val']:
            loss_r_v, loss_v_v = run_model(c['data_type'], model, dataloader_val, train=False, max_samples=1000, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size']*100, max_radius=cn['max_radius'], debug=c['debug'])
        else:
            loss_r_v, loss_v_v = torch.tensor(0.0),torch.tensor(0.0)
        t3 = time.time()

        # Next save results
        loss_t = (loss_r_t + loss_v_t) / 2
        loss_v = (loss_r_v + loss_v_v) / 2

        results = update_results_and_save_to_csv(results, epoch, loss_r_t,loss_v_t,loss_r_v,loss_v_v, csv_file)

        loss = find_relevant_loss(loss_t, loss_t, loss_v, loss_v, c['use_val'], c['loss'])
        if loss < lossBest: # Check if model was better than previous
            lossBest = loss
            epochs_since_best = 0
            torch.save(model.state_dict(), f"{model_name_best}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= c['epochs_for_lr_adjustment']:
                for g in optimizer.param_groups:
                    g['lr'] *= c['lr_adjustment']
                    lr = g['lr']
                epochs_since_best = 0

        LOG.info(f'{epoch:2d}  Loss(train): {loss_t:.2e}  Loss(val): {loss_v:.2e}  Loss_r(train): {loss_r_t:.2e}  Loss_v(train): {loss_v_t:.2e}  Loss_r(val): {loss_r_v:.2e}  Loss_v(val): {loss_v_v:.2e}  Loss_best(val): {lossBest:.2e}  Lr: {lr:2.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  '
                 f'Time(total) {(time.time() - t0)/3600:.1f}h')
        epoch += 1

    model.load_state_dict(torch.load(model_name_best))  # We load the best performing model
    if c['use_test']:
        t4 = time.time()
        loss_test, lossD_test = run_model(c['data_type'], model, dataloader_test, train=False, max_samples=1e6, optimizer=optimizer, loss_fnc=c['loss'], batch_size=c['batch_size'] * 5, max_radius=cn['max_radius'], debug=c['debug'])
        LOG.info(f'Testing...  Loss: {loss_test:.2e}  LossD: {lossD_test:.2e} Time(test) {time.time() - t4:.1f}s')

    if c['viz']:
        plot_training_and_validation(results,c['result_dir'])

    if c['perform_endstep_MD_propagation']:
        assert c['data_type'] == 'water'
        run_model_MD_propagation_simulation(model, dataloader_endstep, log=LOG,viz=c['result_dir'])

    close_logger(LOG)
    # print(f"{torch.cuda.memory_allocated() / 1024 / 1024:2.2f}MB")
    # print(f"{torch.cuda.max_memory_allocated() / 1024 / 1024:2.2f}MB")
    # print(torch.cuda.memory_summary())
    return results, dataloader_train, dataloader_val, dataloader_test, dataloader_endstep
