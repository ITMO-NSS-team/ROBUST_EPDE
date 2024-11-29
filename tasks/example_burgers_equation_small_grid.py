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

    mesh = 100
    path = "data/burgers_equation_small_grid/"
    df = pd.read_csv(f'{path}burgers_sln_{mesh}.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    derives = None

    x = np.linspace(-1000, 0, mesh + 1)
    t = np.linspace(0, 1, mesh + 1)
    grid = np.meshgrid(t, x, indexing='ij')

    params = [x, t]
    mode = "mat" # "mat", "NN"
    domain = Domain()
    domain.variable('x', [-1000, 0], mesh)
    domain.variable('t', [0, 1], mesh)

    boundaries = Conditions()

    # Initial conditions at t=0
    bndval1 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval1.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': [-1000, 0], 't': 0}, value=bndval1)

    # Initial conditions at t=1
    bndval1_2 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval1_2.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': [-1000, 0], 't': 1}, value=bndval1_2)

    # Boundary conditions at x=-1000
    bndval2 = torch.from_numpy(
        pd.read_csv(f'{path}boundary_conditions/burgers_bndval2.csv', header=None).values).reshape(-1).float()

    boundaries.dirichlet({'x': -1000, 't': [0, 1]}, value=bndval2)

    noise = False
    variance_arr = [0.01] if noise else [0]

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
            "boundary": 10,  #
            "verbose_params": {"show_iter_idx": False}
        },
        # "set_memory_properties": {
        #     "mem_for_cache_frac": 10
        # },
        "set_moeadd_params": {
            "population_size": 4,  #
            "training_epochs": 200
        },
        # "Cache_stored_tokens": {
        #     "token_type": "grid",
        #     "token_labels": ["t", "x"],
        #     "params_ranges": {"power": (1, 1)},
        #     "params_equality_ranges": None
        # },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (2, 1),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
            "eq_sparsity_interval": (1e-8, 1e-1), # (1e-8, 5.0),  #
            # "deriv_method": "ANN",  #
            # "deriv_method_kwargs": {"epochs_max": 1000},  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {'smooth': True, 'sigma': 1, 'polynomial_window': 5, 'poly_order': 4},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "results": {
            "level_num": 1 # 2
        },
        "glob_epde": {
            "test_iter_limit": 50,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False, # False,
            "load_result": True # True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "nets": "continuous",  # "discrete", # "continuous",
            "n_bins": 3,
            "sample_k": 35,
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
            "list_unique": ['du/dx0{power: 1.0}_r_u', 'du/dx1{power: 1.0} * u{power: 1.0}_u']
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
            # 'required_bc_ord': (2, 2)
        },
        "Cache": {
            "use_cache": False,
            "save_always": False,
        },
        "StopCriterion": {
            "print_every": 500
        },
        "Optimizer": {
            "learning_rate": 10,
            "lambda_bound": 5,
            "epochs": 3000
        },
        "Plot": {
            "step_plot_print": 500,
            "step_plot_save": 2500,
            "image_save_dir": img_dir,
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries
