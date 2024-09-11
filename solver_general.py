import os
import pandas as pd
import numpy as np
import torch
import sys
import time
import dill as pickle

from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.solver import grid_format_prepare
from tedeous.device import solver_device
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch
from func.transition_bs import text_form_of_equation
from func import transition_bs as transform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')


def solver_equations(cfg, grid, params_full, b_conds, equations, epde_obj: EpdeSearch = False, title=None):
    # solver_device('cuda')
    torch.set_default_dtype(torch.float32)
    if not (os.path.exists(f'data/{title}/solver_result')):
        os.mkdir(f'data/{title}/solver_result')

    dim = cfg.params["global_config"]["dimensionality"] + 1  # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
    k_variable_names = len(cfg.params["fit"]["variable_names"])

    if not b_conds and k_variable_names == 1:
        equation_temp = equations[0]  # Any equation for boundary_conditions
        text_form = text_form_of_equation(equation_temp, cfg)
        eq_g = translate_equation(text_form, epde_obj.pool)
        b_conds = eq_g.boundary_conditions(cfg.params["glob_solver"]["required_bc_ord"], full_domain=True)
        principal_bcond_shape = b_conds[0][1].shape
        # if grid is only square, else !error!
        for i in range(len(b_conds)):
            b_conds[i][1] = b_conds[i][1].reshape(principal_bcond_shape)

    grid_full = grid_format_prepare(params_full, cfg.params["glob_solver"]["mode"]).to('cpu').float()

    set_solutions, models = [], []

    for equation in equations:
        start = time.time()
        eq_s = transform.solver_view(equation, cfg)
        equation = Equation(grid, eq_s, b_conds, h=cfg.params["NN"]["h"]).set_strategy(cfg.params["glob_solver"]["mode"])

        if cfg.params["glob_solver"]["mode"] == 'mat':
            model = torch.rand(grid[0].shape)
            model_arch = torch.nn.Sequential(
                torch.nn.Linear(dim, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, k_variable_names)
            )
        else:     # for variant mode = "NN" and "autograd"
            model = torch.nn.Sequential(
                torch.nn.Linear(dim, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, k_variable_names)
            )

        # opt_model = torch.compile(model)
        model = Solver(grid, equation, model, cfg.params["glob_solver"]["mode"]).solve(
            lambda_bound=cfg.params['Optimizer']['lambda_bound'],
            cache_dir=cfg.params['Cache']['cache_dir'],
            cache_verbose=cfg.params['Cache']['cache_verbose'],
            save_always=cfg.params['Cache']['save_always'],
            use_cache=cfg.params['Cache']['use_cache'],
            model_randomize_parameter=cfg.params['Cache']['model_randomize_parameter'],
            verbose=cfg.params['Verbose']['verbose'],
            learning_rate=cfg.params['Optimizer']['learning_rate'],
            print_every=cfg.params['Verbose']['print_every'],
            no_improvement_patience=cfg.params['StopCriterion']['no_improvement_patience'],
            patience=cfg.params['StopCriterion']['patience'],
            eps=cfg.params['StopCriterion']['eps'], tmin=cfg.params['StopCriterion']['tmin'],
            tmax=cfg.params['StopCriterion']['tmax'],
            cache_model=None if cfg.params["glob_solver"]["mode"] != "mat" else model_arch,
            step_plot_print=cfg.params["Plot"]["step_plot_print"],
            step_plot_save=cfg.params["Plot"]["step_plot_save"],
            image_save_dir=cfg.params["Plot"]["image_save_dir"]) #,
            # backend=True)

        end = time.time()
        print(f'Time = {end - start}')

        Solver(grid, equation, model, cfg.params["glob_solver"]["mode"]).solution_print(solution_print=False, solution_save=True, save_dir=cfg.params["Plot"]["image_save_dir"])
        model_main = Solver(grid, equation, model, cfg.params["glob_solver"]["mode"])
        models.append(model_main)

        solution_function = model if cfg.params["glob_solver"]["mode"] == "mat" else model(grid_full)

        solution_function = solution_function.reshape(*[len(i) for i in params_full]).detach().cpu().numpy() if dim > 1 else solution_function.detach().cpu().numpy()

        if not len(set_solutions):
            set_solutions = [solution_function]
        else:
            set_solutions.append(solution_function)
        # To save temporary solutions
        torch.save(np.array(set_solutions),
                   f'data/{title}/solver_result/file_u_main_{list(np.array(set_solutions).shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    set_solutions = np.array(set_solutions)

    number_of_files = int(len(os.listdir(path=f"data/{title}/solver_result/")))
    if os.path.exists(f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt'):
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}_{number_of_files}.pt')
    else:
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    # # Load data
    # set_solutions, models = torch.load(f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt'), None

    return set_solutions, models
