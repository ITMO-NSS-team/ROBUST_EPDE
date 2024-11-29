import os
import numpy as np
import pandas as pd
import torch
import scipy

import json
from tedeous import config
from default_configs import DEFAULT_CONFIG_EBS
from tedeous.data import Domain, Conditions

config.default_config = json.loads(DEFAULT_CONFIG_EBS)


# equations = [{'du/dx1{power: 1.0} * u{power: 1.0}': -1,
#               'du/dx0{power: 1.0}_r': -1}]  # for burgers_equation or burgers_equation_small_grid

def load_data():
    """
        Load data from github
        https://github.com/urban-fasel/EnsembleSINDy
        https://github.com/urban-fasel/EnsembleSINDy/blob/main/PDE-FIND/datasets/burgers.mat
    """
    path = "data/burgers_equation/"
    mat = scipy.io.loadmat(f'{path}burgers.mat')

    data = mat['u']
    data = np.transpose(data)
    t = np.ravel(mat['t'])
    x = np.ravel(mat['x'])

    dx = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dx_256.csv', header=None)
    d_x = dx.values
    d_x = np.transpose(d_x)

    dt = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dt_256.csv', header=None)
    d_t = dt.values
    d_t = np.transpose(d_t)

    dtt = pd.read_csv(f'{path}wolfram_sln_derv/burgers_sln_dtt_256.csv', header=None)
    d_tt = dtt.values
    d_tt = np.transpose(d_tt)

    derives = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    derives[:, :, 0] = d_t
    derives[:, :, 1] = d_tt
    derives[:, :, 2] = d_x

    # derives = np.zeros(shape=(data.shape[0], data.shape[1], 2))
    # derives[:, :, 0] = d_t
    # derives[:, :, 1] = d_x

    # Create mesh
    grid = np.meshgrid(t, x, indexing='ij')

    params = [x, t]
    mode = "mat" # "mat", "NN"
    domain = Domain()
    domain.variable('x', [-4000, 4000], len(x) - 1)
    domain.variable('t', [0, 4], len(t) - 1)

    boundaries = Conditions()

    # Initial conditions at t=0
    bndval1 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval1.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': [-4000, 4000], 't': 0}, value=bndval1)

    # Initial conditions at t=4
    bndval1_2 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval1_2.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': [-4000, 4000], 't': 4}, value=bndval1_2)

    # Boundary conditions at x=-4000
    bndval2 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval2.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': -4000, 't': [0, 4]}, value=bndval2)

    # # Boundary conditions at x=4000
    # bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([4000], dtype=np.float64)), t_c).float() # x_c[-1]
    # # u(4000,t)=0
    # bndval3 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval3.csv', header=None).values).reshape(-1).float()

    noise = False
    variance_arr = [0.001] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": 1,
            "variance_arr": variance_arr,
            "plot_reverse": True
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 0,  #
            "verbose_params": {"show_iter_idx": False}
        },
        "set_moeadd_params": {
            "population_size": 4,  #
            "training_epochs": 100
        },
        # "Cache_stored_tokens": {
        #     "token_type": "grid",
        #     "token_labels": ["t", "x"],
        #     "params_ranges": {"power": (1, 1)},
        #     "params_equality_ranges": None
        # },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (2, 1), # (1, 1), #
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
            "eq_sparsity_interval": (1e-8, 1e-1), # (1e-8, 5.0), #
            "deriv_method": "ANN",  #
            "deriv_method_kwargs": {"epochs_max": 1000},  #
            # "deriv_method": "poly",
            # "deriv_method_kwargs": {'smooth': True},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "results": {
            "level_num": 1
        },
        "glob_epde": {
            "test_iter_limit": 50,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "nets": "continuous",  # "discrete", # "continuous",
            "n_bins": 3,
            "sample_k": 32,
            "lambda": 0.01,
            "plot": False,
            "save_result": False,
            "load_result": True
        },
        "preprocessor": {
            "encoder_boolean": False,
            "discretizer_boolean": True,
            "strategy": "uniform" # "quantile"
        },
        "correct_structures": {
            "list_unique": ['du/dx0{power: 1.0}_u', 'du/dx1{power: 1.0} * u{power: 1.0}_u']
        }
    }

    img_dir = f'{path}burgers_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": mode,
            "reverse": True,
            "load_result": True
        },
        "StopCriterion": {
            "print_every": 500
        },
        # "Cache": {
        #     "cache_verbose": False,
        #     "use_cache": True,
        #     "save_always": F,
        #     "cache_dir": f"{path}cache/"
        # },
        "Optimizer": {
            "learning_rate": 10,
            "lambda_bound": 5,
            "epochs": 3000
        },
        "Plot": {
            "step_plot_print": 500,
            "step_plot_save": None,
            "image_save_dir": img_dir,
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries

