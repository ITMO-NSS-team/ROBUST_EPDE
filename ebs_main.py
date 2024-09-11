import dill as pickle
import pandas as pd
import numpy as np
import torch
import os

from func.load_data import *

from epde_general import epde_equations
from bamt_general import bs_experiment
from solver_general import solver_equations
from func import confidence_region as conf_plt

if __name__ == '__main__':

    tasks = {
        'wave_equation': wave_equation,  # 0
        'burgers_equation': burgers_equation,  # 1
    }

    title = list(tasks.keys())[1]  # name of the problem (equation/system)

    data, data_grid, derivatives, cfg, solver_grid, params_full, b_conds = tasks[title].load_data()

    for variance in cfg.params["global_config"]["variance_arr"]:

        df_main, epde_search_obj = epde_equations(data, data_grid, derivatives, cfg, variance, title)

        equations = bs_experiment(df_main, cfg, title)

        set_solutions, models = solver_equations(cfg, solver_grid, params_full, b_conds, equations, epde_search_obj, title)

        conf_plt.confidence_region_print(data, data_grid, cfg, params_full, set_solutions, variance, title)
