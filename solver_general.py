import numpy as np
import torch
import sys
import os
import time

from tedeous.data import Equation
from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device
from tedeous.models import mat_model
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from func.transition_bs import text_form_of_equation
from func import transition_bs as transform
import tkinter as tk
from tkinter import filedialog, messagebox

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('../')


def load_result(title):
    root = tk.Tk()
    root.withdraw()
    solver_result_dir = f'data/{title}/solver_result'

    while True:
        try:
            file_path = filedialog.askopenfilename(
                initialdir=solver_result_dir,
                title="Select file",
                filetypes=(("PyTorch files", "*.pt"), ("all files", "*.*"))
            )
            root.withdraw()

            if file_path:
                set_solutions = torch.load(file_path)
                print(f"The file '{file_path}' has been successfully uploaded.")
                root.destroy()
                return set_solutions
            else:
                print("File selection was cancelled.")
                return None
        except Exception as e:
            print(f"Error during file upload: {e}")
            retry = messagebox.askretrycancel("File Load Error", f"Failed to load file: {e}\nRetry?")
            if not retry:
                root.destroy()
                return None


def solver_equations(cfg, domain, params_full, b_conds, equations, epde_obj: EpdeSearch = False, title=None):
    # solver_device('cuda')
    torch.set_default_dtype(torch.float32)
    if not (os.path.exists(f'data/{title}/solver_result')):
        os.mkdir(f'data/{title}/solver_result')

    if cfg.params["glob_solver"]["load_result"]:
        return load_result(title)

    dim = cfg.params["global_config"]["dimensionality"] + 1  # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
    k_variable_names = len(cfg.params["fit"]["variable_names"])

    # if not b_conds and k_variable_names == 1:
    #     equation_temp = equations[0]  # Any equation for boundary_conditions
    #     text_form = text_form_of_equation(equation_temp, cfg)
    #     eq_g = translate_equation(text_form, epde_obj.pool)
    #     b_conds = eq_g.boundary_conditions(cfg.params["glob_solver"]["required_bc_ord"], full_domain=True)
    #     principal_bcond_shape = b_conds[0][1].shape
    #     # if grid is only square, else !error!
    #     for i in range(len(b_conds)):
    #         b_conds[i][1] = b_conds[i][1].reshape(principal_bcond_shape)

    set_solutions = []

    for equation_i in equations:
        start = time.time()
        eq_solver = transform.solver_view(equation_i, cfg)

        equation = Equation()

        if k_variable_names > 1:  # if the system, when we get the list from transform.solver_view
            for eq_i in eq_solver:
                equation.add(eq_i)
        else:
            equation.add(eq_solver)

        if cfg.params["glob_solver"]["mode"] == 'mat':
            net = mat_model(domain, equation)
        else:  # for variant mode = "NN" and "autograd"

            net = torch.nn.Sequential(
                torch.nn.Linear(dim, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, 100),
                torch.nn.Tanh(),
                torch.nn.Linear(100, k_variable_names)
            )

        model = Model(net, domain, equation, b_conds)

        model.compile(mode=cfg.params["glob_solver"]["mode"],
                      lambda_operator=cfg.params['Optimizer']['lambda_operator'],
                      lambda_bound=cfg.params['Optimizer']['lambda_bound'])

        cb_es = early_stopping.EarlyStopping(eps=cfg.params['StopCriterion']['eps'],
                                             no_improvement_patience=cfg.params['StopCriterion']['no_improvement_patience'],
                                             patience=cfg.params['StopCriterion']['patience'],
                                             verbose=cfg.params['StopCriterion']['verbose'],
                                             info_string_every=cfg.params['StopCriterion']['print_every'])

        cb_cache = cache.Cache(cache_dir=cfg.params['Cache']['cache_dir'],
                               cache_verbose=cfg.params['Cache']['cache_verbose'],
                               model_randomize_parameter=cfg.params['Cache']['model_randomize_parameter'])

        cb_plots = plot.Plots(save_every=cfg.params["Plot"]["step_plot_save"],
                              print_every=cfg.params["Plot"]["step_plot_print"],
                              img_dir=cfg.params["Plot"]["image_save_dir"])

        optimizer = Optimizer(optimizer=cfg.params['Optimizer']['optimizer'],
                              params={'lr': cfg.params['Optimizer']['learning_rate']})

        model.train(optimizer, epochs=cfg.params['Optimizer']['epochs'], save_model=cfg.params['Cache']['save_always'], callbacks=[cb_es, cb_plots, cb_cache])

        end = time.time()
        print(f'Time = {end - start}')

        grid = domain.build(cfg.params["glob_solver"]["mode"])

        solution_function = net if cfg.params["glob_solver"]["mode"] == "mat" else net(grid)

        solution_function = solution_function.reshape(*[len(i) for i in params_full]).detach().cpu().numpy() if dim > 1 else solution_function.detach().cpu().numpy()

        if not len(set_solutions):
            set_solutions = [solution_function]
        else:
            set_solutions.append(solution_function)
        # To save temporary solutions
        torch.save(np.array(set_solutions), f'data/{title}/solver_result/file_u_main_{list(np.array(set_solutions).shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    set_solutions = np.array(set_solutions)

    number_of_files = int(len(os.listdir(path=f"data/{title}/solver_result/")))
    if os.path.exists(f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt'):
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}_{number_of_files}.pt')
    else:
        torch.save(set_solutions, f'data/{title}/solver_result/file_u_main_{list(set_solutions.shape)}_{cfg.params["glob_solver"]["mode"]}_{cfg.params["global_config"]["variance_arr"]}.pt')

    return set_solutions
