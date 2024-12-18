import math
import numpy as np
import torch
import plotly.graph_objs as go
import plotly.io as pio
import statistics

from tedeous.device import solver_device
pio.renderers.default = "browser"


def get_rms(records):
    """
        Root-mean-square (rms)
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def confidence_region_print(data, domain, cfg, param, set_solutions, equations, variance, title):
    if set_solutions is None:
        return None

    variable_names = cfg.params["fit"]["variable_names"]

    mean_arr = np.zeros((set_solutions.shape[1], set_solutions.shape[2]))
    var_arr = np.zeros((set_solutions.shape[1], set_solutions.shape[2]))
    s_g_arr = np.zeros((set_solutions.shape[1], set_solutions.shape[2]))  # population standard deviation of data.
    s_arr = np.zeros((set_solutions.shape[1], set_solutions.shape[2]))  # sample standard deviation of data

    for i in range(set_solutions.shape[1]):
        for j in range(set_solutions.shape[2]):
            mean_arr[i, j] = np.mean(set_solutions[:, i, j])
            var_arr[i, j] = np.var(set_solutions[:, i, j])
            s_arr[i, j] = statistics.stdev(set_solutions[:, i, j])

    mean_tens = torch.from_numpy(mean_arr)
    var_tens = torch.from_numpy(var_arr)
    s_g_arr = torch.from_numpy(var_arr) ** (1 / 2)
    s_arr = torch.from_numpy(s_arr)

    # Confidence region for the mean
    upper_bound = mean_tens + 1.96 * s_arr / math.sqrt(len(set_solutions))
    lower_bound = mean_tens - 1.96 * s_arr / math.sqrt(len(set_solutions))

    # case: if dimensionality = 0 - [t, ] and variable_names = [u, v] or [u, v, ..]
    if cfg.params["global_config"]["dimensionality"] == 0:

        prepared_grid_main = domain.build(cfg.params["glob_solver"]["mode"])
        x = prepared_grid_main[0].reshape(-1) if cfg.params["glob_solver"]["mode"] == 'mat' else prepared_grid_main[:, 0]

        for n, solution in enumerate(set_solutions):
            print(f'{n + 1}. equations = {equations[n]}')
            for i, var in enumerate(variable_names):
                error_rmse = np.sqrt(np.mean((data[i].reshape(-1) - solution[:, i].reshape(-1)) ** 2))
                print(f'rmse_{var} = {error_rmse}')
            print('--------------------------')

        confidence_region = torch.cat((upper_bound, torch.flip(lower_bound, dims=(0,))), 0)
        confidence_grid = torch.cat((x, torch.flip(x, dims=(0,))), 0)

        fig = go.Figure()
        for n, param in enumerate(cfg.params["fit"]["variable_names"]):
            color = list(np.random.choice(range(256), size=3))
            fig.add_trace(go.Scatter(x=x, y=data[n].reshape(-1), name=f'Initial field - {param}',
                                     line=dict(color='firebrick', width=4)))

            fig.add_trace(go.Scatter(
                x=x, y=mean_tens[:, n], name=f'Solution field (mean) - {param}',
                line_color=f'rgb({color[0]},{color[1]},{color[2]})',
                line=dict(dash='dash')))

            fig.add_trace(go.Scatter(
                x=confidence_grid,
                y=confidence_region[:, n],
                fill='toself',
                fillcolor=f'rgba({color[0]},{color[1]},{color[2]},0.2)',
                line_color='rgba(255,255,255,0)',
                name=f'Confidence region - {param}',
            ))

        fig.update_layout(title=title,
                          xaxis_title="Time t, [days]",
                          yaxis_title="Population")

        fig.show()

    # case: if dimensionality = 1 - [t, x] and variable_names = [u,]
    if cfg.params["global_config"]["dimensionality"] == 1:
        prepared_grid_main = domain.build(cfg.params["glob_solver"]["mode"])

        if cfg.params["global_config"]["plot_reverse"]:  # important! relationship of parameters between EPDE and SOLVER
            data = np.transpose(data)

        for k in range(len(set_solutions)):
            error_rmse = np.sqrt(np.mean((data.reshape(-1) - set_solutions[k].reshape(-1)) ** 2))
            print(f'{k + 1}. rmse = {error_rmse}, equation = {equations[k]}')

        mean_tens = mean_tens.reshape(-1)
        upper_bound = upper_bound.reshape(-1)
        lower_bound = lower_bound.reshape(-1)

        # building 3-dimensional graph
        x, y = [], []
        if cfg.params["glob_solver"]["mode"] == 'mat':
            x, y = prepared_grid_main[0].reshape(-1), prepared_grid_main[1].reshape(-1)
        else:
            x, y = prepared_grid_main[:, 0],  prepared_grid_main[:, 1]

        fig = go.Figure(data=[
            go.Mesh3d(x=x, y=y, z=mean_tens,
                      name='Solution field',
                      legendgroup='s', showlegend=True, color='lightpink',
                      opacity=1),
            go.Mesh3d(x=x, y=y, z=upper_bound,
                      name='Confidence region',
                      legendgroup='c', showlegend=True, color='blue',
                      opacity=0.20),
            go.Mesh3d(x=x, y=y, z=lower_bound,
                      name='Confidence region',
                      legendgroup='c', color='blue', opacity=0.20),
            go.Mesh3d(x=x, y=y, z=torch.from_numpy(data).reshape(-1),
                      name='Initial field',
                      legendgroup='i', showlegend=True, color='rgb(139,224,164)',
                      opacity=0.5)
        ])

        # if variance:
        #     noise = []
        #     for i in range(data.shape[0]):
        #         noise.append(np.random.normal(0, variance * get_rms(data[i, :]), data.shape[1]))
        #     noise = np.array(noise)
        #     fig.add_trace(go.Mesh3d(x=grid[0].reshape(-1), y=grid[1].reshape(-1), z=torch.from_numpy(data + noise).reshape(-1),
        #                             name='Initial field + noise',
        #                             legendgroup='i_n', showlegend=True, color='rgb(139,224,80)',
        #                             opacity=0.5))

        fig.update_layout(scene_aspectmode='auto')
        fig.update_layout(showlegend=True,
                          scene=dict(
                              xaxis_title='x1',
                              yaxis_title='x2',
                              zaxis_title=variable_names[0],
                              aspectratio={"x": 1, "y": 1, "z": 1}
                          ),
                          height=800, width=800
                          )
        fig.show()

        fig = go.Figure(data=go.Contour(x=x, y=y, z=mean_tens, contours_coloring='heatmap'))
        fig.update_layout(title_text='Visualization of the equation solution')
        fig.show()

        fig = go.Figure(data=go.Contour(x=x, y=y, z=var_tens.reshape(-1), contours_coloring='heatmap'))
        fig.update_layout(title_text='Visualization of the variance')
        fig.show()
