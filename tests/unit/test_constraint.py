import os
import random

import pytest
import torch
from omegaconf import OmegaConf

from src.constraints import generate_constraints, ConstraintTemplate
from src.optimization import generate_constraints_minimizer, MinimizationTemplate


@pytest.fixture
def constraint_config_path():
    path = "./../config/constraint"
    return path

@pytest.fixture
def minimization_config_path():
    path = "./../config/minimization"
    return path


def fill_missing_values(cfg):
    for key in cfg.keys():
        if key not in cfg:
            cfg[key] = random.randint(1,10)
    return cfg


def test_constraint_yaml(constraint_config_path):
    """
    This test checks that the yaml that exist in the constraint folder are applicable to the constraint class,
    and that all resulting constraint functions are subclasses of the ConstraintTemplate.
    """
    for filename in os.listdir(constraint_config_path):
        if filename.endswith(".yaml"):
            file = os.path.join(constraint_config_path, filename)
            cfg = OmegaConf.load(file)
            cfg = fill_missing_values(cfg)
            con_fnc = generate_constraints(cfg)
            assert isinstance(con_fnc, ConstraintTemplate)

def test_minimization_yaml(constraint_config_path,minimization_config_path):
    """
    This test checks that the yaml that exist in the minimization folder are applicable with the minimization class,
    and that all resulting minimization functions are subclasses of the MinimizationTemplate.
    """
    for filename in os.listdir(constraint_config_path):
        if filename.endswith(".yaml"):
            file = os.path.join(constraint_config_path, filename)
            cfg_con = OmegaConf.load(file)
            cfg_con = fill_missing_values(cfg_con)
            con_fnc = generate_constraints(cfg_con)
            for filename in os.listdir(minimization_config_path):
                if filename.endswith(".yaml"):
                    file = os.path.join(minimization_config_path, filename)
                    cfg_min = OmegaConf.load(file)
                    cfg_min = fill_missing_values(cfg_min)
                    min_con_fnc = generate_constraints_minimizer(cfg_min,con_fnc=con_fnc)
                    assert isinstance(min_con_fnc, MinimizationTemplate)



def test_constraint_consitency(constraint_config_path,minimization_config_path):
    """
    We test that the constraints actually takes data of the shape they are supposed to.
    Furthermore we test that the constraint violation is smaller
    #TODO We also need to test that the different methods are working (cg, steepest descent, batch)
    #TODO Check that the second order constraint is also running without error and that it is lowering.
    #TODO We need to test the penalty function as well
    #TODO Finally we need to check that the function works in high dimension.

    """
    n_batch = 10
    for filename in os.listdir(constraint_config_path):
        if filename.endswith(".yaml"):
            file = os.path.join(constraint_config_path, filename)
            cfg_con = OmegaConf.load(file)
            cfg_con = fill_missing_values(cfg_con)
            con_fnc = generate_constraints(cfg_con)
            for filename in os.listdir(minimization_config_path):
                if filename.endswith(".yaml"):
                    file = os.path.join(minimization_config_path, filename)
                    cfg_min = OmegaConf.load(file)
                    cfg_min = fill_missing_values(cfg_min)
                    min_con_fnc = generate_constraints_minimizer(cfg_min,con_fnc=con_fnc)

                    pos = con_fnc.position_idx
                    vel = con_fnc.velocity_idx
                    ncon = random.randint(1, 10)
                    dim_in = len(pos) + len(vel)
                    x = torch.randn(n_batch, ncon, dim_in)
                    x_con = min_con_fnc(x)
                    cv_after, cv_mean_after, cv_max_after = con_fnc.constraint_violation(x_con)
                    cv, cv_mean, cv_max = con_fnc.constraint_violation(x)
                    assert (cv_mean_after < cv_mean) and (cv_max_after < cv_max)
