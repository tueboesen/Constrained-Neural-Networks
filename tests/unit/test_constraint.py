import random

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import os
import pytest
from src.constraints import generate_constraints, ConstraintTemplate


@pytest.fixture
def config_path():
    path = "./../../config/constraint"
    return path


def test_constraint_yaml(config_path):
    """
    This test checks that the yaml that exist in the constraint folder are applicable to the constraint class,
    and that all resulting constraint functions are subclasses of the ConstraintTemplate.
    """
    for filename in os.listdir(config_path):
        if filename.endswith(".yaml"):
            file = os.path.join(config_path,filename)
            cfg = OmegaConf.load(file)
            con_fnc = generate_constraints(cfg)
            assert isinstance(con_fnc, ConstraintTemplate)

def test_constraint_consitency(config_path):
    """
    We test that the constraints actually takes data of the shape they are supposed to.
    Furthermore we test that the constraint violation is smaller
    #TODO We also need to test that the different methods are working (cg, steepest descent, batch)
    #TODO Check that the second order constraint is also running without error and that it is lowering.
    #TODO Finally we need to check that the function works in high dimension.
    #TODO We need to test the penalty function as well

    """
    n_batch = 10
    for filename in os.listdir(config_path):
        if filename.endswith(".yaml"):
            file = os.path.join(config_path,filename)
            cfg = OmegaConf.load(file)
            con_fnc = generate_constraints(cfg)
            pos = con_fnc.position_idx
            vel = con_fnc.velocity_idx
            if con_fnc.n_constraints is None:
                ncon = random.randint(1,10)
            else:
                ncon = con_fnc.n_constraints
            dim_in = len(pos)+len(vel)
            x = torch.randn(n_batch,ncon,dim_in)
            x_con, reg, reg2 = con_fnc(x)
            cv_after, cv_mean_after, cv_max_after = con_fnc.constraint_violation(x_con)
            cv, cv_mean, cv_max = con_fnc.constraint_violation(x)
            assert (cv_mean_after < cv_mean) and (cv_max_after < cv_max)

