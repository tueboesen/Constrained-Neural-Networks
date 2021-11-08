import copy
import os
from datetime import datetime

import numpy as np

from src.main import main
from src.vizualization import plot_training_and_validation_accumulated
from e3nn import o3


def standard_network_sizes(c,network_type):
    """
    These are the standard networks that are used in this project
    """

    if network_type.lower() == 'eq':
        c['network'] = {
            'irreps_inout': o3.Irreps("6x1o"),
            'irreps_hidden': o3.Irreps("30x0o+30x0e+20x1o+20x1e"),
            'layers': 8,
            'max_radius': 15,
            'number_of_basis': 8,
            'embed_dim': 2,
            'max_atom_types': 20,
            'radial_neurons': [48],
            'num_neighbors': -1,
        }
    elif network_type.lower() == 'mim':
        c['network'] = {
           'node_dim_in': 9,
            'node_dim_latent': 120,
            'nlayers': 8,
            'max_radius': 15,
        }
    else:
        raise NotImplementedError("Network type not recognized")


    return c

def create_job(c,network_type, con,con_type,seed,use_same_data,jobid,repetition):
    """
    Creates a job to be run.
    """
    c['con'] = con
    c['con_type'] = con_type
    c['seed'] = seed
    c['repetition'] = repetition
    c['jobid'] = jobid
    c['network_type'] = network_type
    if not ('network' in c):
        c = standard_network_sizes(c,network_type)
    if c['loss'] == '':
        c['loss'] = c['network_type']
    if use_same_data and jobid > 0:
        c['use_same_data'] = True
    else:
        c['use_same_data'] = False
    c['result_dir'] = f"{c['result_dir_base']}/{c['network_type']}_{c['con']}_{c['con_type']}_{repetition}/"
    legend = f"{c['con']:} {c['con_type']:}"
    jobid = jobid + 1
    return c,jobid,legend



def job_planner(c):
    """
    This function can plan out multiple jobs to be run
    """
    c['result_dir_base'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )
    os.makedirs(c['result_dir_base'])
    c_base = copy.deepcopy(c)
    cs = []
    legends = []
    for i,seed in enumerate(c_base['seed']):
        jobid = 0
        for network_type in c_base['network_type']:
            for con in c_base['con']:
                if con == '':
                    con_type = 'low'
                    c_new,jobid,legend = create_job(copy.deepcopy(c),network_type,con,con_type,seed,c_base['use_same_data'],jobid,i)
                    cs.append(c_new)
                    if i == 0:
                        legends.append(legend)
                else:
                    for con_type in c_base['con_type']:
                        c_new, jobid, legend = create_job(copy.deepcopy(c), network_type, con, con_type, seed, c_base['use_same_data'], jobid,i)
                        cs.append(c_new)
                        if i == 0:
                            legends.append(legend)
    results = np.zeros((len(legends),len(c_base['seed']),4,c['epochs']))
    return cs, legends, results




def job_runner(cs,legends, results):
    """
    This function runs a set of jobs, stores the results and plot the training and validation loss
    """
    dataloader_train = None
    dataloader_val = None
    dataloader_test = None
    dataloader_endstep = None
    for i,c in enumerate(cs):
        if c['use_same_data'] == False:
            dataloader_train = None
            dataloader_val = None
            dataloader_test = None
            dataloader_endstep = None
        result,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep = main(c,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep)

        results[c['jobid'],c['repetition'],0,:] = result['loss_t']
        results[c['jobid'],c['repetition'],1,:] = result['loss_v']
        results[c['jobid'],c['repetition'],2,:] = result['lossD_t']
        results[c['jobid'],c['repetition'],3,:] = result['lossD_v']
        file = f"{c['result_dir_base']:}/results"
        np.save(file,results)
        plot_training_and_validation_accumulated(results, legends, c['result_dir_base'])
        plot_training_and_validation_accumulated(results, legends, c['result_dir_base'],semilogy=True)

    return