import os
import pandas as pd
import numpy as np
import re
import itertools
import dill as pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt

import bamt.networks as nets
import bamt.nodes as nodes
import bamt.preprocessors as pp
from pgmpy.estimators import K2Score


def token_check(columns, objects_res, config_bamt):
    list_correct_structures_unique = config_bamt.params["correct_structures"]["list_unique"]
    variable_names = config_bamt.params["fit"]["variable_names"]

    list_correct_structures = set()
    for term in list_correct_structures_unique:
        # str_r = '_r' if '_r' in term else ''
        str_elem = ''
        if any(f'_{elem}' in term for elem in variable_names):
            for elem in variable_names:
                if f'_{elem}' in term:
                    term = term.replace(f'_{elem}', "")
                    str_elem = f'_{elem}'
        # for case if several terms exist
        arr_term = re.sub('_r', '', term).split(' * ')
        perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_elem # + str_r
            list_correct_structures.add(temp)

    def out_red(text):
        print("\033[31m {}".format(text), end='')

    def out_green(text):
        print("\033[32m {}".format(text), end='')

    met, k_sys = 0, len(objects_res)
    k_min = k_sys if k_sys < 5 else 5

    for object_row in objects_res[:k_min]:
        k_c, k_l = 0, 0
        for col in columns:
            if col in object_row:
                col_temp = re.sub('_r', '', col)
                if col_temp in list_correct_structures:
                    k_c += 1
                    out_green(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
                else:
                    k_l += 1
                    out_red(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
        print(f'correct structures = {k_c}/{len(list_correct_structures_unique)}')
        print(f'incorrect structures = {k_l}')
        print('--------------------------')

    for object_row in objects_res:
        for temp in object_row.keys():
            if temp in list_correct_structures:
                met += 1

    print(f'average value (equation - {k_sys}) = {met / k_sys}')


def get_objects(synth_data, config_bamt):
    """
        Parameters
        ----------
        synth_data : pd.dataframe
            The fields in the table are structures of received systems/equations,
            where each record/row contains coefficients at each structure
        config_bamt:  class Config from TEDEouS/config.py contains the initial configuration of the task

        Returns
        -------
        objects_result - list objects (combination of equations or systems)
    """
    objects = []  # equations or systems
    for i in range(len(synth_data)):
        object_row = {}
        for col in synth_data.columns:
            object_row[synth_data[col].name] = synth_data[col].values[i]
        objects.append(object_row)

    objects_result = []
    for i in range(len(synth_data)):
        object_res = {}
        for key, value in objects[i].items():
            if abs(float(value)) > config_bamt.params["glob_bamt"]["lambda"]:
                object_res[key] = value

        if len(object_res) >= 2:
            objects_result.append(object_res)
        # print(f'{i + 1}.{object_res}')  # full string  output
        # print('--------------------------')

    return objects_result


def bs_experiment(df, cfg, title):

    path = f'data/{title}/bamt_result'
    if not (os.path.exists(path)):
        os.mkdir(path)

    if cfg.params["glob_bamt"]["load_result"]:
        with open(f'{path}/data_equations_{cfg.params["glob_bamt"]["sample_k"]}.pickle', 'rb') as f:
            objects_res = pickle.load(f)
            if cfg.params["correct_structures"]["list_unique"] is not None:
                token_check(df.columns, objects_res, cfg)
            return objects_res

    # Rounding values
    for col in df.columns:
        df[col] = df[col].round(decimals=16)
    # Deleting rows with condition
    df = df.loc[(df.sum(axis=1) != -len(cfg.params["fit"]["variable_names"])), (df.sum(axis=0) != 0)]
    # Deleting null columns
    df = df.loc[:, (df != 0).any(axis=0)]
    # (df != 0).sum(axis = 0) # Number of non-zero values in columns/structures

    df_temp = (df[[col for col in df.columns if
                   'C' not in col]] != 0).copy()  # exclude the free term column for calculations
    init_nodes_list = []
    for i in range(len(cfg.params["fit"]["variable_names"])):
        init_nodes = df_temp.sum(axis=0).idxmax()
        init_nodes_list.append(init_nodes)
        df_temp = df_temp.drop(init_nodes, axis=1)

    if cfg.params["glob_bamt"]["nets"] == 'Continuous':
        df_main = df.copy()

        bn = nets.ContinuousBN(use_mixture=True)

        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(n_bins=cfg.params["glob_bamt"]["n_bins"], encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
        discrete_data, est = p.apply(df_main)
        info = p.info


    else: # when the table includes discrete values (right part -1 or 0)
        # An array to keep track of if a row has been changed
        modified_rows = [False] * len(df)

        for init_node in init_nodes_list:
            for i in range(len(df)):
                if not modified_rows[i]:  # Check if the row has been changed before
                    value = df[init_node].iloc[i]
                    if value not in [0, -1]:
                        factor = - df[init_node].iloc[i]
                        df.iloc[i, :] = df.iloc[i, :] / factor  # Divide the whole string by the opposite value to get -1
                        modified_rows[i] = True

        modified_count = sum(modified_rows)
        print(f"Number of rows modified: {modified_count}")

        df_main = df.copy()

        for node_i in init_nodes_list:
            df_main.rename(columns={node_i: node_i[:-2] + '_r_' + node_i[-1]}, inplace=True)

        for col in df_main.columns:
            if '_r' in col:
                df_main = df_main.astype({col: "int64"})
                df_main = df_main.astype({col: "str"})

        all_r = df_main.shape[0]
        unique_r = df_main.groupby(df.columns.tolist(), as_index=False).size().shape[0]

        print(f'Out of the {all_r} equations obtained, \033[1m {unique_r} unique \033[0m ({int(unique_r / all_r * 100)} %)')

        l_r, l_left = [], []
        for term in list(df.columns):
            if '_r' in term:
                l_r.append(term)
            else:
                l_left.append(term)
        df_main = df_main[l_left + l_r]

        bn = nets.HybridBN(has_logit=True, use_mixture=True)

        discretizer = preprocessing.KBinsDiscretizer(n_bins=cfg.params["glob_bamt"]["n_bins"], encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('discretizer', discretizer)])
        discrete_data, est = p.apply(df_main)
        info = p.info

    bn.add_nodes(info)

    params = {"init_nodes": init_nodes_list} if not cfg.params["params"]["init_nodes"] else cfg.params[
        "params"]

    bn.add_edges(discrete_data, scoring_function=('K2', K2Score), params=params)
    bn.plot(f'output_main_{title}.html') # , f'{path}/'
    bn.fit_parameters(df_main)

    objects_res = []
    limit = 10
    while len(objects_res) < cfg.params["glob_bamt"]["sample_k"] and limit > 0:
        synth_data = bn.sample(cfg.params["glob_bamt"]["sample_k"], as_df=True)
        temp_res = get_objects(synth_data, cfg)

        if len(temp_res) + len(objects_res) > cfg.params["glob_bamt"]["sample_k"]:
            objects_res += temp_res[:cfg.params["glob_bamt"]["sample_k"] - len(objects_res)]
        else:
            objects_res += temp_res

        limit -= 1

    if cfg.params["correct_structures"]["list_unique"] is not None:
        token_check(df.columns, objects_res, cfg)

    if cfg.params["glob_bamt"]["save_result"]:
        number_of_files = len(os.listdir(path=f"data/{title}/bamt_result/"))
        if os.path.exists(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.csv'):
            with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}_{number_of_files}.pickle', 'wb') as f:
                pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'data/{title}/bamt_result/data_equations_{len(objects_res)}.pickle', 'wb') as f:
                pickle.dump(objects_res, f, pickle.HIGHEST_PROTOCOL)

    return objects_res
