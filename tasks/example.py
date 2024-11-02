
import os
import sys
import time

import math
import numpy as np
import pandas as pd
import torch

import json
from tedeous import config
from default_configs import DEFAULT_CONFIG_EBS
from tedeous.data import Domain, Conditions


def load_data():
    """
        path -> data -> parameters -> derivatives (optional) -> grid -> boundary conditions (optional) -> modules config (optional)
    """
    path = """YOUR CODE HERE"""
    data = """YOUR CODE HERE"""

    derives = None  # if there are no derivatives

    grid = """YOUR CODE HERE"""
    domain = """YOUR CODE HERE"""
    params = """YOUR CODE HERE"""

    boundaries = False  # if there are no boundary conditions

    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-frienly 
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)

    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

    Meaning c1*u*d2u/dx2 has the form

    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}

    None is for function without derivatives

    bval=torch.Tensor prescribed values at every point in the boundary

    bconds = [[bnd1, bop1, bndval1], [bnd2, bop2, bndval2], ...]
    """

    noise = False
    variance_arr = ["""YOUR CODE HERE"""] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": data.ndim - 1 # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
        }
    }

    epde_config = {"""YOUR CODE HERE"""}

    bamt_config = {"""YOUR CODE HERE"""}

    solver_config = {"""YOUR CODE HERE"""}

    config_modules = {**global_modules,
                      **epde_config,
                      **bamt_config,
                      **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries


def filter_condition(SoEq, variable_names) -> bool:
    """
    Filter for quality and complexity of equations/system when collecting equations from the Pareto front
    SoEq.obj_fun[:len(variable_names)] - quality
    SoEq.obj_fun[len(variable_names):] - complexity
    SoEq.obj_fun - quality and complexity list
    variable_names = SoEq.vals.equation_keys # params for object_epde_search
    Parameters
    ----------
    SoEq: an object of the class 'epde.structure.main_structures.SoEq'
    variable_names: List of objective function names
    """
    pass
