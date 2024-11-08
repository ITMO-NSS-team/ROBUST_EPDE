
from tasks import example_wave_equation, example_burgers_equation

from epde_general import epde_equations
from bamt_general import bs_experiment
from solver_general import solver_equations

from func import confidence_region as conf_plt

if __name__ == '__main__':

    tasks = {
        'wave_equation': example_wave_equation,  # 0
        'burgers_equation': example_burgers_equation,  # 1
    }

    title = list(tasks.keys())[0]  # name of the problem (equation/system)

    data, data_grid, derivatives, cfg, domain, params_full, b_conds = tasks[title].load_data()

    for variance in cfg.params["global_config"]["variance_arr"]:

        df_main, epde_search_obj = epde_equations(data, data_grid, derivatives, cfg, variance, title)

        equations = bs_experiment(df_main, cfg, title)

        set_solutions = solver_equations(cfg, domain, params_full, b_conds, equations, epde_search_obj, title)

        conf_plt.confidence_region_print(data, domain, cfg, params_full, set_solutions, equations, variance, title)
