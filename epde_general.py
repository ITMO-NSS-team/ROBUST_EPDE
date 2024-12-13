import copy
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg

from func import obj_collection as collection


def equation_definition(grid, config_epde, title):
    selected_module = __import__(f'tasks.example_{title}', fromlist=[''])

    dimensionality = config_epde.params["global_config"]["dimensionality"]

    deriv_method_kwargs = {}
    if config_epde.params["fit"]["deriv_method"] == "poly":
        deriv_method_kwargs = {'use_smoothing': config_epde.params["fit"]["deriv_method_kwargs"]["smooth"],
                               'sigma': config_epde.params["fit"]["deriv_method_kwargs"]["sigma"],
                               'polynomial_window': config_epde.params["fit"]["deriv_method_kwargs"][
                                   "polynomial_window"],
                               'poly_order': config_epde.params["fit"]["deriv_method_kwargs"]["poly_order"]}
    elif config_epde.params["fit"]["deriv_method"] == "ANN":
        deriv_method_kwargs = {'epochs_max': config_epde.params["fit"]["deriv_method_kwargs"]["epochs_max"]}

    multiobjective_mode = True
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode,
                                          use_solver=config_epde.params["epde_search"]["use_solver"],
                                          dimensionality=dimensionality,
                                          boundary=config_epde.params["epde_search"]["boundary"],
                                          coordinate_tensors=grid,
                                          memory_for_cache=config_epde.params["fit"]["memory_for_cache"],
                                          prune_domain=config_epde.params["fit"]["prune_domain"],
                                          verbose_params=config_epde.params["epde_search"]["verbose_params"],
                                          function_form=None if config_epde.params["epde_search"]["function_form"] is None else getattr(selected_module, 'function_form'))

    epde_search_obj.set_preprocessor(default_preprocessor_type=config_epde.params["fit"]["deriv_method"],
                                     preprocessor_kwargs=deriv_method_kwargs)
    # epde_search_obj.set_memory_properties(data, mem_for_cache_frac=config_epde.params["set_memory_properties"][
    #     "mem_for_cache_frac"])
    epde_search_obj.set_moeadd_params(population_size=config_epde.params["set_moeadd_params"]["population_size"],
                                      training_epochs=config_epde.params["set_moeadd_params"]["training_epochs"])

    add_tokens = [] if config_epde.params["fit"]["additional_tokens"] == [] else getattr(selected_module, 'additional_tokens')()

    return epde_search_obj, add_tokens


def equation_fit(epde_search_obj, data, derives, config_epde, add_tokens):
    """ Method epde_search.fit() is used to initiate the equation search."""

    if len(config_epde.params["fit"]["variable_names"]) < 2 and derives is not None:
        derives = [derives]

    epde_search_obj.fit(data=data, variable_names=config_epde.params["fit"]["variable_names"],
                        data_fun_pow=config_epde.params["fit"]["data_fun_pow"],
                        max_deriv_order=config_epde.params["fit"]["max_deriv_order"],
                        equation_terms_max_number=config_epde.params["fit"]["equation_terms_max_number"],
                        equation_factors_max_number=config_epde.params["fit"]["equation_factors_max_number"],
                        # coordinate_tensors=grid,
                        eq_sparsity_interval=config_epde.params["fit"]["eq_sparsity_interval"],
                        derivs=derives if derives is not None else None,
                        additional_tokens=add_tokens)

    '''
    The results of the equation search have the following format: if we call method 
    .equations with "only_print = True", the Pareto frontiers 
    of equations of varying complexities will be shown, as in the following example:

    If the method is called with the "only_print = False", the algorithm will return list 
    of Pareto frontiers with the desired equations.
    '''
    epde_search_obj.equations(only_print=True, num=config_epde.params["results"]["level_num"])
    config_epde.params["fit"]["init_new_pool"] = True
    return epde_search_obj


def epde_equations(u, grid_u, derives, cfg, variance, title):
    if not (os.path.exists(f'data/{title}/epde_result')):
        os.mkdir(f'data/{title}/epde_result')

    k = 0  # number of equations (final)
    variable_names = cfg.params["fit"]["variable_names"]  # list of objective function names
    table_main = [{i: [{}, {}]} for i in variable_names]  # dict/table coefficients left/right parts of the equation

    # Loading temporary data (for saving temp results)
    if os.path.exists(f'data/{title}/epde_result/table_main_general.pickle'):
        with open(f'data/{title}/epde_result/table_main_general.pickle', 'rb') as f:
            table_main = pickle.load(f)
        with open(f'data/{title}/epde_result/k_main_general.pickle', 'rb') as f:
            k = pickle.load(f)

    epde_obj, add_tokens = equation_definition(grid_u, cfg, title)

    if cfg.params["glob_epde"]["load_result"]:

        if cfg.params["glob_solver"]["type"] == 'odeint':
            epde_obj.create_pool(data=u, variable_names=cfg.params["fit"]["variable_names"],
                                 derivs=derives if derives is not None else None,
                                 max_deriv_order=cfg.params["fit"]["max_deriv_order"],
                                 additional_tokens=add_tokens, data_fun_pow=cfg.params["fit"]["data_fun_pow"])
            # deriv_fun_pow=deriv_fun_pow, data_nn=data_nn, fourier_layers=fourier_layers, fourier_params=fourier_params)

            # Need to check the existence of the file or send the path
            return pd.read_csv(f'data/{title}/epde_result/output_main_{title}.csv', index_col='Unnamed: 0',
                               sep='\t',
                               encoding='utf-8'), epde_obj
        else:
            return pd.read_csv(f'data/{title}/epde_result/output_main_{title}.csv', index_col='Unnamed: 0',
                               sep='\t',
                               encoding='utf-8'), False

    for test_idx in np.arange(cfg.params["glob_epde"]["test_iter_limit"]):
        while True:
            try:
                epde_obj = equation_fit(epde_obj, u, derives, cfg, add_tokens)
                break
            except Exception as e:
                print(f"Error: {e}. Restart algorithm...")

        res = epde_obj.equations(only_print=False, num=cfg.params["results"]["level_num"])  # result search

        table_main, k = collection.object_table(res, variable_names, table_main, k, title)
        # To save temporary data
        with open(f'data/{title}/epde_result/table_main_general.pickle', 'wb') as f:
            pickle.dump(table_main, f, pickle.HIGHEST_PROTOCOL)

        with open(f'data/{title}/epde_result/k_main_general.pickle', 'wb') as f:
            pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)

        print(test_idx)

    frame_main = collection.preprocessing_bamt(variable_names, table_main, k)

    if cfg.params["glob_epde"]["save_result"]:
        if os.path.exists(f'data/{title}/epde_result/output_main_{title}.csv'):
            frame_main.to_csv(
                f'data/{title}/epde_result/output_main_{title}_{len(os.listdir(path=f"data/{title}/epde_result/"))}.csv',
                sep='\t', encoding='utf-8')
        else:
            frame_main.to_csv(f'data/{title}/epde_result/output_main_{title}.csv', sep='\t', encoding='utf-8')

    return frame_main, epde_obj
