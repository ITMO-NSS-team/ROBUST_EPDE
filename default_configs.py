DEFAULT_CONFIG_EBS = '''
{
"global_config": {
"discovery_module": "EPDE",
"dimensionality": 2,
"variance_arr": [0],
"plot_reverse": false
},
"EPDE_config": "commentary",
"epde_search": {
"use_solver": false,
"boundary": 0,
"verbose_params": {"show_iter_idx": true},
"function_form": null
},
"set_memory_properties": {
"mem_for_cache_frac": 10
},
"set_moeadd_params": {
"population_size": 10,
"training_epochs": 5
},
"Cache_stored_tokens": {
"token_type": "grid",
"token_labels": ["t", "x"],
"params_ranges": {"power": [1, 1]},
"params_equality_ranges": null
},
"fit": {
"variable_names": ["u"],
"max_deriv_order": [2, 2],
"equation_terms_max_number": 3,
"data_fun_pow": 1,
"additional_tokens": [],
"equation_factors_max_number": 1,
"eq_sparsity_interval": [1e-4, 2.5],
"derivs": null,
"deriv_method": "poly",
"deriv_method_kwargs": {"smooth": false, "sigma": 1, "polynomial_window": 5, "poly_order": 4},
"memory_for_cache": 5,
"prune_domain": false,
"init_new_pool":false
},
"results":{
"level_num": 1
},
"glob_epde": {
"test_iter_limit": 1,
"save_result": true,
"load_result": false
},

"BAMT_config": "commentary",
"glob_bamt": {
"nets": "continuous",
"n_bins": 10,
"sample_k": 35,
"lambda": 0.001,
"plot": false,
"save_result": false,
"load_result": false
},
"preprocessor":{
"strategy": "quantile",
"encoder_boolean": true,
"discretizer_boolean": true
},
"params": {
"init_nodes": false
},
"correct_structures":{
"list_unique": null
},

"SOLVER_config": "commentary",
"glob_solver": {
"mode": "NN",
"reverse": false,
"required_bc_ord": [2, 2],
"load_result": false
},
"Optimizer": {
"learning_rate":1e-4,
"lambda_bound":10,
"lambda_operator":1,
"optimizer":"Adam",
"epochs":5e6
},
"Cache":{
"use_cache":false,
"cache_dir":"../cache/",
"cache_verbose":false,
"save_always":false,
"model_randomize_parameter":0
},
"NN":{
"batch_size":null,
"lp_par":null,
"grid_point_subset":["central"],
"h":0.001
},
"StopCriterion":{
"eps":1e-5,
"tmin":1000,
"tmax":1e5 ,
"patience":5,
"loss_oscillation_window":100,
"no_improvement_patience":1000,
"verbose":true,
"print_every":null   	
},
"Matrix":{
"lp_par":null,
"cache_model":null
},
"Plot":{
"step_plot_print":null,
"step_plot_save":null, 
"image_save_dir":null
}
}
'''