#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from tedeous import config
from default_configs import DEFAULT_CONFIG_EBS
from tedeous.data import Domain, Conditions

config.default_config = json.loads(DEFAULT_CONFIG_EBS)
from epde.interface.interface import EpdeSearch

from epde.supplementary import define_derivatives
from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

import matplotlib.pyplot as plt
import matplotlib

# SMALL_SIZE = 12
# matplotlib.rc('font', size=SMALL_SIZE)
# matplotlib.rc('axes', titlesize=SMALL_SIZE)


import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

from pathlib import Path

if __name__ == "__main__":

    root_path = Path("..")
    print(root_path.resolve())  # Абсолютный путь
    path = root_path.resolve()

    # df_smooth = pd.read_csv(f'{path}/data_real/hudson-bay-lynx-hare-smooth.csv', sep=';', engine='python')
    # data_smooth = df_smooth.values
    # t = data_smooth[:, 0]
    # x = data_smooth[:, 1]  # Lynx/Hunters - рысь
    # y = data_smooth[:, 2]  # Hare/Prey - заяц
    # data = [y, x]

    # t_min, t_max = 0, 20
    # t_interval = f"t_({t_min}, {t_max})"
    # t = np.linspace(t_min, t_max, len(t))


    data_initial = np.load(f'{path}/data_synth/data_synth.npy')
    t = np.load(f'{path}/data_synth/t_synth.npy')
    x = data_initial.T[:, 0] # Hare
    y = data_initial.T[:, 1] # Lynx
    data = [x, y]

    grid = [t, ]


    # derivatives_u = np.load(f'{path}/data_real/derivatives_u.npy')
    # derivatives_v = np.load(f'{path}/data_real/derivatives_v.npy')
    #
    # du_dt = derivatives_u[:, 0]
    # dv_dt = derivatives_v[:, 0]

    # derives_u = np.zeros(shape=(derivatives_u.shape[0], 1))
    # derives_u[:, 0] = du_dt
    #
    # derives_v = np.zeros(shape=(derivatives_v.shape[0], 1))
    # derives_v[:, 0] = dv_dt
    #
    # derives = [derives_u, derives_v]

    derivatives_u_synth = np.load(f'{path}/data_synth/derivatives_u_synth.npy')[::]
    derivatives_v_synth = np.load(f'{path}/data_synth/derivatives_v_synth.npy')[::]

    du_dt = derivatives_u_synth[:, 0]
    # du_dtt = derivatives_u_synth[:, 1]

    dv_dt = derivatives_v_synth[:, 0]
    # dv_dtt = derivatives_v_synth[:, 1]

    default_preprocessor_type = 'poly'
    preprocessor_kwargs = {'use_smoothing': True, 'sigma': 1, 'polynomial_window': 5, 'poly_order': 4}

    # default_preprocessor_type = 'ANN'
    # preprocessor_kwargs = {'epochs_max': 10000}

    setup = PreprocessorSetup()
    builder = ConcretePrepBuilder()
    setup.builder = builder

    if default_preprocessor_type == 'ANN':
        setup.build_ANN_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'poly':
        setup.build_poly_diff_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'spectral':
        setup.build_spectral_preprocessing(**preprocessor_kwargs)
    else:
        raise NotImplementedError('Incorrect default preprocessor type. Only ANN or poly are allowed.')
    preprocessor_pipeline = setup.builder.prep_pipeline

    if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
        preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None

    max_order = (1, )

    deriv_names, deriv_orders = define_derivatives('u', dimensionality=1, max_order=max_order)

    data_tensor, derivatives_u = preprocessor_pipeline.run(x, grid=grid, max_order=max_order)

    deriv_names_v, deriv_orders_v = define_derivatives('v', dimensionality=1, max_order=max_order)

    data_tensor_v, derivatives_v = preprocessor_pipeline.run(y, grid=grid, max_order=max_order)

    dx_dt_v2 = np.gradient(x, t)
    dy_dt_v2 = np.gradient(y, t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name=f'Координата X'))
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=f'Координата Y'))

    fig.add_trace(go.Scatter(x=t, y=derivatives_u.reshape(-1), mode='lines',
                             name=f'Производные (epde) X'))
    fig.add_trace(go.Scatter(x=t, y=derivatives_v.reshape(-1), mode='lines',
                             name=f'Производные (epde) Y'))

    fig.add_trace(go.Scatter(x=t, y=dx_dt_v2.reshape(-1), mode='lines',
                             name=f'Производные (np.gradient) X'))
    fig.add_trace(go.Scatter(x=t, y=dy_dt_v2.reshape(-1), mode='lines',
                             name=f'Производные (np.gradient) Y'))


    fig.add_trace(go.Scatter(x=t, y=du_dt.reshape(-1), mode='lines',
                             name=f'Производные загруженные X'))
    fig.add_trace(go.Scatter(x=t, y=dv_dt.reshape(-1), mode='lines',
                             name=f'Производные загруженные Y'))

    fig.update_xaxes(title="Итерации")
    fig.update_yaxes(title="Значения")
    fig.update_layout(title=f'Маятник и его производные')
    fig.show()

    # np.save(f'derivatives_lotka_volterra_{default_preprocessor_type}_{t_interval}.npy', [derivatives_u, derivatives_v])
