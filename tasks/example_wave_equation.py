import os
import numpy as np
import pandas as pd
import torch
import json
from tedeous import config
from default_configs import DEFAULT_CONFIG_EBS
from tedeous.data import Domain, Conditions

config.default_config = json.loads(DEFAULT_CONFIG_EBS)


# equations = [{'d^2u/dx1^2{power: 1.0}_u': 0.04, 'd^2u/dx0^2{power: 1.0}_u': -1.0}] # for wave_equation ([t-0,x-1]-epde, [x-0, t-1]-solver)
# set_solutions = torch.load(
#     f'data/{title}/solver_result/file_u_main_[35, 71, 71]_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')



def load_data():
    """
         Load data
         Synthetic data from wolfram:

         WE = {D[u[x, t], {t, 2}] - 1/25 ( D[u[x, t], {x, 2}]) == 0}
         bc = {u[0, t] == 0, u[1, t] == 0};
         ic = {u[x, 0] == 10000 Sin[1/10 x (x - 1)]^2, Evaluate[D[u[x, t], t] /. t -> 0] == 1000 Sin[1/10  x (x - 1)]^2}
         NDSolve[Flatten[{WE, bc, ic}], u, {x, 0, 1}, {t, 0, 1}]
     """

    mesh = 70

    path = 'data/wave_equation/'
    df = pd.read_csv(f'{path}wolfram_sln/wave_sln_{mesh}.csv', header=None)
    data = df.values
    data = np.transpose(data)  # x1 - t (axis Y), x2 - x (axis X)

    derives = None

    t = np.linspace(0, 1, mesh + 1)
    x = np.linspace(0, 1, mesh + 1)
    grid = np.meshgrid(t, x, indexing='ij')

    params = [x, t]
    mode = "autograd"
    domain = Domain()
    domain.variable('x', [0, 1], mesh)
    domain.variable('t', [0, 1], mesh)


    boundaries = False  # if there are no boundary conditions
    """
    Preparing boundary conditions (BC)
    
    bnd=torch.Tensor of a boundary n-D points where n is the problem dimensionality
    
    bop=dictionary in the form of 
    'С * u{power: 1.0} * d^2u/dx1^2{power: 1.0}': 
            {
                'coeff': С ,
                'vars_set': [[None], [0, 0]],
                'power_set': [1.0, 1.0] 
            },
    
    bndval=torch.Tensor prescribed values at every point in the boundary
    """

    boundaries = Conditions()

    x_s = domain.variable_dict['x']

    # Initial conditions at t=0, u(0,x)= 10000*sin[1/10 x*(x - 1)]^2
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=10000 * torch.sin(0.1 * x_s * (x_s - 1)) ** 2)

    # Initial conditions at t=0, du/dt = 1000*sin[1/10 x*(x - 1)]^2
    # d/dt
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1,
                'var': 0
            }
    }

    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=1000 * torch.sin(0.1 * x_s * (x_s - 1)) ** 2)


    # Boundary conditions at x=0, u(0,t)=0
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=0)

    # Boundary conditions at x=1, u(1,t)=0
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=0)


    noise = False
    variance_arr = [0.001] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": 1, # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
            "variance_arr": variance_arr,
            "plot_reverse": True
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 15,  #
            "verbose_params": {"show_iter_idx": False}
        },
        "set_moeadd_params": {
            "population_size": 5,  #
            "training_epochs": 100
        },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (2, 2),
            "equation_terms_max_number": 4,  #
            # "equation_factors_max_number": {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
            "equation_factors_max_number": 1,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            "deriv_method": "poly",
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "glob_epde": {
            "test_iter_limit": 100,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "nets": "continuous", # "discrete", # "continuous",
            "n_bins": 5,
            "sample_k": 35,
            "lambda": 0.01,
            "plot": False,
            "save_result": False,
            "load_result": True
        },
        "correct_structures": {
            "list_unique": ['d^2u/dx1^2{power: 1.0}_u', 'd^2u/dx0^2{power: 1.0}_u']
        }
    }

    img_dir = f'{path}wave_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": mode,
            "reverse": True,
            "load_result": True
        },
        "Cache": {
            "use_cache": False,
            "save_always": False,
            "cache_dir": f"{path}cache/"
        },
        "StopCriterion": {
            "print_every": 500
        },
        "Optimizer": {
            "learning_rate": 1e-3,
            "lambda_bound": 100,
            "optimizer": "Adam",
            "epochs": 1500
        },
        "Plot": {
            "step_plot_print": 500,
            "step_plot_save": None,
            "image_save_dir": img_dir,
        }
    }

    ebs_config = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries


# def filter_condition(SoEq, variable_names): # filter the quality and the complexity of equations/system
#     if all([v < 5 for v in SoEq.obj_fun[:len(variable_names)]]): # and (max(SoEq.obj_fun[len(variable_names):]) < 4):  # to filter the quality and the complexity of equations/system
#         return True
#     return False
