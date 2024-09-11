import math
import os
import numpy as np
import pandas as pd
import torch
import scipy.io
from scipy.ndimage import gaussian_filter
import json
from tedeous import config
from default_configs import DEFAULT_CONFIG_EBS
from torch.cuda.amp import autocast
from epde.interface.prepared_tokens import ExternalDerivativesTokens
from tedeous.solver import grid_format_prepare

config.default_config = json.loads(DEFAULT_CONFIG_EBS)


def example_equation():
    """
        path -> data -> parameters -> derivatives (optional) -> grid -> boundary conditions (optional) -> modules config (optional)
    """
    path = """YOUR CODE HERE"""
    data = """YOUR CODE HERE"""

    derives = None  # if there are no derivatives

    grid = """YOUR CODE HERE"""
    param = """YOUR CODE HERE"""

    bconds = False  # if there are no boundary conditions

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

    return data, grid, derives, cfg_ebs, param, bconds


def wave_equation():
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

    mode = "NN"
    grid_solver = grid_format_prepare(params, mode).to('cpu').float()

    bconds = False  # if there are no boundary conditions
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

    x_c = torch.from_numpy(x)
    t_c = torch.from_numpy(t)

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    bop1 = None

    # u(0,x)= 10000*sin[1/10 x*(x - 1)]^2
    bndval1 = (10000 * torch.sin((0.1 * bnd1[:, 0] * (bnd1[:, 0] - 1)) ** 2)).float()

    # Initial conditions at t=0
    bnd2 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()

    # d/dt
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dt': [1],
                'pow': 1
            }
    }

    # du/dt = 1000*sin[1/10 x*(x - 1)]^2
    bndval2 = (1000 * torch.sin((0.1 * bnd2[:, 0] * (bnd2[:, 0] - 1)) ** 2)).float()

    # Boundary conditions at x=0
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t_c).float()

    bop3 = None

    # u(0,t)=0
    bndval3 = torch.from_numpy(np.zeros(len(bnd3), dtype=np.float64))

    # Boundary conditions at x=1
    bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t_c).float()

    bop4 = None
    # u(1,t)=0
    bndval4 = torch.from_numpy(np.zeros(len(bnd4), dtype=np.float64)).float()

    # Putting all bconds together
    bconds = [[bnd1, bndval1, 'dirichlet'], [bnd2, bop2, bndval2, 'operator'], [bnd3, bndval3, 'dirichlet'], [bnd4, bndval4, 'dirichlet']]
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
            "boundary": 15,  #
            "verbose_params": {"show_iter_idx": False}
        },
        # "set_memory_properties": {
        #     "mem_for_cache_frac": 10
        # },
        "set_moeadd_params": {
            "population_size": 5,  #
            "training_epochs": 20
        },
        # "Cache_stored_tokens": {
        #     "token_type": "grid",
        #     "token_labels": ["t", "x"],
        #     "params_ranges": {"power": (1, 1)},
        #     "params_equality_ranges": None
        # },
        "fit": {
            "variable_names": ['u', ],
            "max_deriv_order": (2, 2),
            "equation_terms_max_number": 3,  #
            "equation_factors_max_number": 1,
            "eq_sparsity_interval": (1e-8, 5.0),  #
            "deriv_method": "poly",
            # "deriv_method_kwargs": {"smooth": True},
            "memory_for_cache": 25,
            "prune_domain": False
        },
        "glob_epde": {
            "test_iter_limit": 10,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": False
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 35,
            "lambda": 0.0001,
            "plot": False,
            "save_result": False,
            "load_result": True
        },
        "params": {
            "init_nodes": ['d^2u/dx2^2{power: 1.0}']
        },
        "correct_structures": {
            "list_unique": ['d^2u/dx2^2{power: 1.0}', 'd^2u/dx1^2{power: 1.0}_r']
        }
    }

    img_dir = f'{path}wave_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": mode,
            "reverse": True
        },
        "Cache": {
            "use_cache": True,
            "save_always": False,
            "cache_dir": f"{path}cache/"
        },
        "Optimizer": {
            "learning_rate": 1e-3,
            "lambda_bound": 100,
            "optimizer": "Adam"
        },
        "Plot": {
            "step_plot_print": False,
            "step_plot_save": False,
            "image_save_dir": img_dir,
        }
    }

    ebs_config = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}ebs_config.json', 'w') as fp:
        json.dump(ebs_config, fp)

    cfg_ebs = config.Config(f'{path}ebs_config.json')

    # return data, grid, derives, cfg_ebs, param, bconds
    return data, grid, derives, cfg_ebs, grid_solver, params, bconds


def burgers_equation():
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

    derives = None
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

    mode = "mat"
    grid_solver = grid_format_prepare(params, mode).to('cpu').float()

    bconds = False  # if there are no boundary conditions
    x_c = torch.from_numpy(x)
    t_c = torch.from_numpy(t)

    # Initial conditions at t=0
    bnd1 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([0], dtype=np.float64))).float()
    # u(x, 0) = Piecewise
    bndval1 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval1.csv', header=None).values).reshape(-1).float()

    # Initial conditions at t=4
    bnd2 = torch.cartesian_prod(x_c, torch.from_numpy(np.array([4], dtype=np.float64))).float()
    # u(x, 4) = Piecewise
    bndval2 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval1_2.csv', header=None).values).reshape(-1).float()

    # Boundary conditions at x=-4000
    bnd3 = torch.cartesian_prod(torch.from_numpy(np.array([-4000], dtype=np.float64)), t_c).float() # x_c[0]
    # u(-4000,t)=1000
    bndval3 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval2.csv', header=None).values).reshape(-1).float()

    # # Boundary conditions at x=4000
    # bnd4 = torch.cartesian_prod(torch.from_numpy(np.array([4000], dtype=np.float64)), t_c).float() # x_c[-1]
    # # u(4000,t)=0
    # bndval4 = torch.from_numpy(pd.read_csv(f'{path}boundary_conditions/burgers_bndval3.csv', header=None).values).reshape(-1).float()
    # Putting all bconds together
    # bconds = [[bnd1, bndval1, 'dirichlet'], [bnd2, bndval2, 'dirichlet'], [bnd3, bndval3, 'dirichlet'], [bnd4, bndval4, 'dirichlet']]
    bconds = [[bnd1, bndval1, 'dirichlet'], [bnd2, bndval2, 'dirichlet'], [bnd3, bndval3, 'dirichlet']]
    noise = False
    variance_arr = [0.001] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": 1,
            "variance_arr": variance_arr
        }
    }

    epde_config = {
        "epde_search": {
            "use_solver": False,
            "boundary": 0,  #
            "verbose_params": {"show_iter_idx": False}
        },
        # "set_memory_properties": {
        #     "mem_for_cache_frac": 10
        # },
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
            "test_iter_limit": 25,  # how many times to launch algorithm (one time - 2-3 equations)
            "save_result": False,
            "load_result": True
        }
    }

    bamt_config = {
        "glob_bamt": {
            "sample_k": 35,
            "lambda": 0.0001,
            "plot": False,
            "save_result": False,
            "load_result": True
        },
        "params": {
            "init_nodes": 'du/dx1{power: 1.0}_r'
        },
        "correct_structures": {
            "list_unique": ['du/dx1{power: 1.0}_u', 'du/dx2{power: 1.0} * u{power: 1.0}_u']
        }
    }

    img_dir = f'{path}burgers_img'

    if not (os.path.isdir(img_dir)):
        os.mkdir(img_dir)

    solver_config = {
        "glob_solver": {
            "mode": mode,
            "reverse": True
        },
        "Cache": {
            "use_cache": True,
            "save_always": True,
            "cache_dir": f"{path}cache/"
        },
        "Optimizer": {
            "learning_rate": 100,
            "lambda_bound": 5,
        },
        "Plot": {
            "step_plot_print": False,
            "step_plot_save": False,
            "image_save_dir": img_dir,
        }
    }

    config_modules = {**global_modules, **epde_config, **bamt_config, **solver_config}

    with open(f'{path}config_modules.json', 'w') as fp:
        json.dump(config_modules, fp)

    cfg_ebs = config.Config(f'{path}config_modules.json')

    # return data, grid, derives, cfg_ebs, param, bconds
    return data, grid, derives, cfg_ebs, grid_solver, params, bconds
