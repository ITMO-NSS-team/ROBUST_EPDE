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


# equations = [
#     {'u{power: 1.0}_u': 0.55, 'v{power: 1.0} * u{power: 1.0}_u': -0.028, 'du/dx0{power: 1.0}_u': -1.0,
#      'v{power: 1.0}_v': -0.84, 'v{power: 1.0} * u{power: 1.0}_v': 0.026, 'dv/dx0{power: 1.0}_v': -1.0}]


def load_data():

    path = "data/lotka_volterra_equations/"

    # df_smooth = pd.read_csv(f'{path}data_real/hudson-bay-lynx-hare-smooth.csv', sep=';', engine='python')
    # data_smooth = df_smooth.values
    # t = data_smooth[:, 0]
    # x = data_smooth[:, 1]  # Lynx/Hunters - рысь
    # y = data_smooth[:, 2]  # Hare/Prey - заяц
    # data = [y, x]
    # t = np.linspace(0, 20, len(t))

    data_initial = np.load(f'{path}data_synth/data_synth.npy')
    t = np.load(f'{path}data_synth/t_synth.npy')
    x = data_initial.T[:, 0] # Hare
    y = data_initial.T[:, 1] # Lynx
    data = [x, y]

    grid = [t, ]
    params = [t, ]
    mode = "autograd"

    domain = Domain()
    domain.variable('t', [0, 20], len(t) - 1)

    derives = None

    # derivatives_u, derivatives_v = np.load(f'{path}data_real/derivatives_lotka_volterra_poly_t_(0, 20).npy')
    #
    # du_dt = derivatives_u[:, 0]
    # dv_dt = derivatives_v[:, 0]
    #
    # derives_u = np.zeros(shape=(derivatives_u.shape[0], 1))
    # derives_u[:, 0] = du_dt
    #
    # derives_v = np.zeros(shape=(derivatives_v.shape[0], 1))
    # derives_v[:, 0] = dv_dt
    #
    # derives = [derives_u, derives_v]

    x0, y0 = x[0], y[0]

    boundaries = Conditions()
    boundaries.dirichlet({'t': 0}, value=x0, var=0)
    boundaries.dirichlet({'t': 0}, value=y0, var=1)

    noise = False
    variance_arr = [0.10] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": 0, # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 5,  #
            "verbose_params": {"show_iter_idx": False}
        },
        "set_moeadd_params": {
            "population_size": 7,  #
            "training_epochs": 300
        },
        "fit": {
            "variable_names": ['u', 'v'],  # list of objective function names
            "max_deriv_order": (1,),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.8, 0.2]}, # the amount of tokens in the term and their probability of occurrence
            "data_fun_pow": 1,  # the maximum degree of one token in the term
            "eq_sparsity_interval": (1e-10, 1e-2),  #
            "deriv_method": "poly",
            "deriv_method_kwargs": {'smooth': True, 'sigma': 1, 'polynomial_window': 5, 'poly_order': 4},
            # "deriv_method": "ANN",  #
            # "deriv_method_kwargs": {"epochs_max": 20000},  #
            "prune_domain": False
        },
        "results": {
            "level_num": 1
        },
        "glob_epde": {
            "test_iter_limit": 50,  # how many times to launch algorithm (one time - 2-3 equations)
            # "save_result": True,
            # "load_result": False
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "round": 14,
            "nets": "continuous", # "discrete", # "continuous",
            "n_bins": 6,
            "sample_k": 35,
            "lambda": 0.01,
            "save_result": True,
            "load_result": False,
            # "save_result": False,
            # "load_result": True
        },
        "params": {
            "init_nodes": ['du/dx0{power: 1.0}_u', 'dv/dx0{power: 1.0}_v']
        },
        "preprocessor": {
            "encoder_boolean": False,
            "discretizer_boolean": True,
            "strategy": "uniform" # "quantile", uniform
        },
        "correct_structures": {
            "list_unique": ['v{power: 1.0} * u{power: 1.0}_v', 'v{power: 1.0}_v', 'dv/dx0{power: 1.0}_r_v',
                            'v{power: 1.0} * u{power: 1.0}_u', 'u{power: 1.0}_u', 'du/dx0{power: 1.0}_r_u']
        }
    }

    # img_dir = f'{path}hunter_prey_img'
    #
    # if not (os.path.isdir(img_dir)):
    #     os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "type": 'odeint',
            "mode": mode,  # "NN",
            "reverse": False,
            "load_result": False
        },
        "StopCriterion": {
            "print_every": 500
        },
        # "Cache": {
        #     "use_cache": False, # True,
        #     "save_always": False, # False,
        #     "cache_dir": f"{path}cache/",
        #     "model_randomize_parameter": 1e-5
        # },
        "Optimizer": {
            "learning_rate": 1e-4,
            "lambda_bound": 1000,
            "epochs": 5000
        },
        "NN": {
            "h": 0.00001
        },
        "Plot": {
            "step_plot_print": 500,
            "step_plot_save": 1000
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries


# def filter_condition(SoEq, variable_names): # filter the quality and the complexity of equations/system
#     if all([v < 5 for v in SoEq.obj_fun[:len(variable_names)]]): # and (max(SoEq.obj_fun[len(variable_names):]) < 4):  # to filter the quality and the complexity of equations/system
#         return True
#     return False

