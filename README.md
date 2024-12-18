# Robust-EPDE

---
Program complex for robust training of models in the form of differential equations.
---

### The algorithms used:

- EPDE (Evolutionary Partial Differential Equations). Algorithm for the search of differential equations based on data (https://github.com/ITMO-NSS-team/EPDE)
- BAMT (Bayesian Analytical and Modelling Toolkit). Repository of a data modeling and analysis tool based on Bayesian networks. (https://github.com/aimclub/BAMT)
- TEDEouS (Torch Exhaustive Differential Equation Solver). Algorithm for automated solution of differential equations based on neural networks (https://github.com/ITMO-NSS-team/torch_DE_solver)

![title](docs/EBS_new.png)

---
## Examples & Tutorials

Before starting the program component, the configuration of each module must be set up. This is done for each task separately in the folder `tasks/`. The file should be named `tasks/example_{title}.py`. Where all additional functions are implemented, starting with `def load_data():`
```Python
    def load_data():
    """
        path -> data -> parameters -> derivatives (optional) -> grid -> 
        -> boundary conditions (optional) -> modules config (optional)
    """
    path = """YOUR CODE HERE"""
    data = """YOUR CODE HERE"""

    derives = None  # if there are no derivatives

    grid = """YOUR CODE HERE"""
    params = """YOUR CODE HERE"""
    
    domain = """YOUR CODE HERE"""
    boundaries = False  # if there are no boundary conditions

    noise = False
    variance_arr = ["""YOUR CODE HERE"""] if noise else [0]

    global_modules = {
        "global_config": {
            "discovery_module": "EPDE",
            "dimensionality": # (starts from 0 - [t,], 1 - [t, x], 2 - [t, x, y])
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

    cfg_ebs = config_modules.Config(f'{path}config_modules.json')

    return data, grid, derives, cfg_ebs, domain, params, boundaries
```

The default configuration and corresponding parameters of each module can be viewed in `default_configs.py`. The task instructions are then added to the main `ebs_main.py` file:
```Python
tasks = {
    'wave_equation': example_wave_equation
}

title = list(tasks.keys())[0] # name of the problem/equation
```

The data archive for equation discovery and recovery is [here](https://github.com/6Naci/data-archive-for-equation-discovery-and-recovery).

### Results

<details>
<summary>1. Wave equation with one spatial variable (more) </summary>

```math 
\frac{\partial^{2} u}{\partial t^{2}} - \frac{1}{25} \frac{\partial^{2} u}{\partial x^{2}} = 0,
```
```math 
\\ 100\times100, x \in [0; 1], t \in [0; 1].
```
The output of the `EPDE` module is presented in the form of a table. The fields in the table are the structures of the obtained partial differential equations, where each row contains the coefficients at each structure. 

These data are input to the `BAMT` module to build a Bayesian network based on them. 
The results of the module are: 
* distribution of coefficients at structures (illustrated in the figure below)
![title](docs/examples/wave_equation/distribution.png) 
* list of sampled partial differential equations, which in turn is the input to the `SOLVER` module.

The resulting solution fields of partial differential equations are used to construct a confidence region and an average solution. The results are displayed for comparison with the original data and as heat maps.
![title](docs/examples/wave_equation/figure_all.gif)

![title](docs/examples/wave_equation/heatmap_1.png)
![title](docs/examples/wave_equation/heatmap_2.png)

</details>

<details>
<summary>2. Burgers' equation </summary>
<br>

```math 
\frac{\partial u}{\partial t} +  u \frac{\partial u}{\partial x} = 0,
```
```math 
\\ 256\times256, x \in [-4000; 4000], t \in [0; 4].
```
![title](docs/examples/burgers_equation/distribution.png)

![title](docs/examples/burgers_equation/figure_all.gif)

![title](docs/examples/burgers_equation/heatmap_2.png)
![title](docs/examples/burgers_equation/heatmap_1.png)

</details>

<details>
<summary>3. Burgers' equation (reduced grid) </summary>

```math 
 \frac{\partial u}{\partial t} +  u \frac{\partial u}{\partial x} = 0,
```
```math 
 \\ 100 \times100, x \in [-1000; 0], t \in [0; 1].
```
</details>

<details>
<summary>4. Lotka-Volterra equations </summary>

```math 
 \begin{equation*}
 \begin{cases}
  \frac{\partial u}{\partial t} = \normalsize 0.55 \cdot u - 0.028 \cdot u \cdot v, 
   \\
   \frac{\partial v}{\partial t} = \normalsize - 0.84 \cdot v + 0.026 \cdot u \cdot v.
 \end{cases}
\end{equation*}
```
```math 
 \\ t \in [0, 20],
 \\ u_0, \ v_0 = 30, 4.
```

![title](docs/examples/lotka_volterra_equations/figure_1.png)

</details>

<details>
<summary>5. Pendulum equations </summary>

```math 
    \begin{equation}
    \begin{gathered}
    \begin{cases}
    \dot\sigma  =  z, 
    \\
    \dot z =  -\frac{1}{\sqrt{5}} z -\sin(\sigma) + 0.2 \quad 
    \end{cases}
    \end{gathered}
    \end{equation}
```
```math 
 \\ t \in [0, 20],
 \\ \sigma_0, \ z_0 = \frac{\pi}{2}, 0.5.
```
</details>




---
