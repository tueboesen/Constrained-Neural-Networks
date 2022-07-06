import copy
import os
from datetime import datetime

import numpy as np

from src.main import main
from src.utils import define_data_keys
from src.vizualization import plot_training_and_validation_accumulated, plot_training_and_validation_accumulated_2, plot_training_and_validation_accumulated_3, \
    plot_training_and_validation_accumulated_4
from e3nn import o3


def standard_network_sizes(c,network_type):
    """
    These are the standard networks that are used in this project. The exception is the multi-body pendulum that works in 2D
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

def create_job(c,network_type, con,con_type,seed,use_same_data,jobid,repetition,regularizationparameter='',gamma=1):
    """
    Creates a job to be run.
    """
    c['con'] = con
    c['con_type'] = con_type
    c['seed'] = seed
    c['repetition'] = repetition
    c['jobid'] = jobid
    c['network_type'] = network_type
    c['regularizationparameter'] = regularizationparameter
    c['gamma'] = gamma
    if not ('network' in c):
        c = standard_network_sizes(c,network_type)
    if c['loss'] == '':
        c['loss'] = c['network_type']
    if use_same_data and jobid > 0:
        c['use_same_data'] = True
    else:
        c['use_same_data'] = False
    c['result_dir'] = f"{c['result_dir_base']}/{c['network_type']}_{c['con']}_{c['con_type']}_{c['regularizationparameter']}_{c['gamma']}_{repetition}/"
    legend = f"{network_type} {c['con']:} {c['con_type']:} {c['gamma']}"
    jobid = jobid + 1

    if con == 'angles':
        c['model_specific']['angles'] = True
        c['network']['node_dim_in'] = 1
        c['con'] = ''

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
                if con == '' or con == 'angles':
                    con_type = ''
                    c_new,jobid,legend = create_job(copy.deepcopy(c),network_type,con,con_type,seed,c_base['use_same_data'],jobid,i,1)
                    cs.append(c_new)
                    if i == 0:
                        legends.append(legend)
                else:
                    for con_type in c_base['con_type']:
                        if con_type == 'reg':
                            for regularizationparameter in c_base['regularizationparameter']:
                                c_new, jobid, legend = create_job(copy.deepcopy(c), network_type, con, con_type, seed, c_base['use_same_data'], jobid,i,regularizationparameter)
                                cs.append(c_new)
                                if i == 0:
                                    legends.append(legend)
                        elif con_type == 'stabhigh':
                            for gamma in c_base['gamma']:
                                c_new, jobid, legend = create_job(copy.deepcopy(c), network_type, con, con_type, seed, c_base['use_same_data'], jobid,i,gamma=gamma)
                                cs.append(c_new)
                                if i == 0:
                                    legends.append(legend)

                        else:
                            c_new, jobid, legend = create_job(copy.deepcopy(c), network_type, con, con_type, seed, c_base['use_same_data'], jobid, i, 1)
                            cs.append(c_new)
                            if i == 0:
                                legends.append(legend)

    # results = {}
    # keys = define_data_keys()
    # for key in keys:
    #     results[key] = np.zeros((len(legends),len(c_base['seed']),c['epochs']))
    results = np.zeros((len(legends),len(c_base['seed']),10,c['epochs']))
    return cs, legends, results




def job_planner2(c):
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
        for network_type,con_type,model_file in zip(c_base['network_type'],c_base['con_type'],c_base['load_previous_model_file']):
            c_new,jobid,legend = create_job(copy.deepcopy(c),network_type,c_base['con'],con_type,seed,c_base['use_same_data'],jobid,i,1)
            c_new['load_previous_model_file']= model_file
            cs.append(c_new)
            if i == 0:
                legends.append(legend)
    results = np.zeros((len(legends),len(c_base['seed']),10,c['epochs']))
    return cs, legends, results






def job_runner(cs,legends, results):
    """
    This function runs a set of jobs, stores the results and plot the training and validation loss
    """
    dataloader_train = None
    dataloader_val = None
    dataloader_test = None
    dataloader_endstep = None
    file_legend = f"{cs[0]['result_dir_base']}/legends.txt"

    with open(file_legend, 'w') as f:
        for i, legend in enumerate(legends):
            f.write(f"{i}: {legend} \n")
    for i,c in enumerate(cs):
        if c['use_same_data'] == False:
            dataloader_train = None
            dataloader_val = None
            dataloader_test = None
            dataloader_endstep = None
        else:
            if dataloader_train is not None:
                if c['data_type'] == 'n-pendulum' and c['model_specific']['angles'] == True:
                    dataloader_train.dataset.useprimary = False
                    dataloader_val.dataset.useprimary = False
                else:
                    dataloader_train.dataset.useprimary = True
                    dataloader_val.dataset.useprimary = True

        result,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep = main(c,dataloader_train,dataloader_val,dataloader_test,dataloader_endstep)

        # for (key,val) in result.items():
        #     results[key][c['jobid'],c['repetition'],:] = val
        results[c['jobid'],c['repetition'],0,:] = result['loss_r_t']
        results[c['jobid'],c['repetition'],1,:] = result['loss_r_v']
        results[c['jobid'],c['repetition'],2,:] = result['loss_v_t']
        results[c['jobid'],c['repetition'],3,:] = result['loss_v_v']
        results[c['jobid'],c['repetition'],4,:] = result['cv_t']
        results[c['jobid'],c['repetition'],5,:] = result['cv_v']
        results[c['jobid'],c['repetition'],6,:] = result['cv_max_t']
        results[c['jobid'],c['repetition'],7,:] = result['cv_max_v']
        results[c['jobid'],c['repetition'],8,:] = result['MAE_r_t']
        results[c['jobid'],c['repetition'],9,:] = result['MAE_r_v']

        file = f"{c['result_dir_base']:}/results"
        np.save(file,results)
        # plot_training_and_validation_accumulated_2(results, legends, c['result_dir_base'])
        plot_training_and_validation_accumulated_4(results, legends, c['result_dir_base'],semilogy=True)
    return