import io
import numpy as np
import pandas as pd
from typing import Callable
from matplotlib import pyplot as plt
from scipy import integrate
from typing import TypedDict, Optional

BOUNDARY_CONVECTIVE = "convective"
BOUNDARY_CONSTANT = "constant temperature"
BOUNDARY_INSULATED = "insulated"


class Config(TypedDict):
    thermal_conductivity: float
    heat_capacity: float
    density: float
    grid_spacing_meters: float
    number_of_grid_nodes: int
    simulation_time_seconds: float
    boundary_condition_start: str
    boundary_condition_end: str
    boundary_temperature_start: Optional[float]
    boundary_temperature_end: Optional[float]
    heat_trans_coeff_start: Optional[float]
    heat_trans_coeff_end: Optional[float]
    starting_temperature: float


class SolverConfig(TypedDict):
    solver_physics: Callable
    starting_temperatures: list


class ResolveInitialTemperatures:
    """Configure the boundary conditions."""
    @staticmethod
    def resolve(setup: Config):
        if setup["number_of_grid_nodes"] > 20:
            raise Warning("Number of grid nodes too high")

        temperatures = np.array([setup["starting_temperature"]] *
                                setup["number_of_grid_nodes"])

        if setup["boundary_condition_start"] == BOUNDARY_CONSTANT:
            temperatures[0] = setup["boundary_temperature_start"]

        if setup["boundary_condition_end"] == BOUNDARY_CONSTANT:
            temperatures[-1] = setup["boundary_temperature_end"]

        return temperatures


class Physics:
    def __init__(self, setup: Config):
        self._boundary_start_map = {
            BOUNDARY_CONSTANT: self._contant_temp,
            BOUNDARY_INSULATED: self._insulated_start,
            BOUNDARY_CONVECTIVE: self._convective_start,
        }

        self._boundary_end_map = {
            BOUNDARY_CONSTANT: self._contant_temp,
            BOUNDARY_INSULATED: self._insulated_end,
            BOUNDARY_CONVECTIVE: self._convective_end,
        }
        self._boundary_start = self._boundary_start_map[setup["boundary_condition_start"]]
        self._boundary_end = self._boundary_end_map[setup["boundary_condition_end"]]

    def __call__(self, t, temperatures, config: Config):
        constant = config["thermal_conductivity"] / (config["heat_capacity"] *
                                                     config["density"])
        gradients = np.zeros(len(temperatures))
        gradients[1:-1] = (constant *
                           (temperatures[:-2] - 2 * temperatures[1:-1] + temperatures[2:]) /
                           config["grid_spacing_meters"]**2)

        self._boundary_start(gradients, temperatures, config)
        self._boundary_end(gradients, temperatures, config)
        return gradients

    @staticmethod
    def _contant_temp(gradients, temperatures, config: Config):
        # Does not need configuring
        pass

    @staticmethod
    def _convective_start(gradients, temperatures, config: Config):
        h = config["heat_trans_coeff_start"]
        k = config["thermal_conductivity"]
        T_infinity = config["boundary_temperature_start"]
        c = 1 / (config["density"] * config["heat_capacity"] * config["grid_spacing_meters"])
        gradients[0] = (c *
                        (h * (T_infinity - temperatures[0]) + k *
                         (temperatures[2] - temperatures[1]) / config["grid_spacing_meters"]))

    @staticmethod
    def _convective_end(gradients, temperatures, config: Config):
        h = config["heat_trans_coeff_end"]
        k = config["thermal_conductivity"]
        T_infinity = config["boundary_temperature_end"]
        c = 1 / (config["density"] * config["heat_capacity"] * config["grid_spacing_meters"])
        gradients[-1] = (
            c * (h * (T_infinity - temperatures[-1]) + k *
                 (temperatures[-3] - temperatures[-2]) / config["grid_spacing_meters"]))

    @staticmethod
    def _insulated_end(gradients, temperatures, config: Config):
        constant = config["thermal_conductivity"] / (config["heat_capacity"] *
                                                     config["density"])

        gradients[-1] = (constant * (temperatures[-2] - temperatures[-1]) /
                         config["grid_spacing_meters"]**2)

    @staticmethod
    def _insulated_start(gradients, temperatures, config: Config):
        constant = config["thermal_conductivity"] / (config["heat_capacity"] *
                                                     config["density"])

        gradients[0] = -(constant * (temperatures[0] - temperatures[1]) /
                         config["grid_spacing_meters"]**2)


_RETURN_CODES = {
    2:
    "Integration successful.",
    -1:
    "Excess work done on this call. (Perhaps wrong MF.)",
    -2:
    "Excess accuracy requested. (Tolerances too small.)",
    -3:
    "Illegal input detected. (See printed message.)",
    -4:
    "Repeated error test failures. (Check all input.)",
    -5:
    "Repeated convergence failures. (Perhaps bad Jacobian supplied or wrong choice of MF or tolerances.)",
    -6:
    "Error weight became zero during problem. (Solution component i vanished, and ATOL or ATOL(i) = 0.)",
}


class Solver:
    @staticmethod
    def solve(solver_config: SolverConfig, config: Config):
        temperatures = []
        times = []
        temperatures_initial = solver_config["starting_temperatures"]

        temperatures.append(temperatures_initial)
        times.append(0.0)

        t_end = config["simulation_time_seconds"]
        time_step = t_end / 100.0
        ode_obj = integrate.ode(solver_config["solver_physics"])
        ode_obj.set_f_params(config)

        ode_obj.set_initial_value(temperatures_initial, 0.0)
        ode_obj.set_integrator("vode", method="bdf", rtol=1.0e-12)

        while ode_obj.successful() and ode_obj.t < t_end:
            new_time = ode_obj.t + time_step
            result = ode_obj.integrate(new_time)
            times.append(new_time)
            temperatures.append(result)

        if not ode_obj.successful():
            return_code = ode_obj.get_return_code()
            raise Warning(f"ODE solver failed with return code: {return_code},"
                          f" with message: {_RETURN_CODES.get(return_code, '')}")

        return times, temperatures


def get_solver_config(setup: Config) -> SolverConfig:
    initial_temperatures = ResolveInitialTemperatures().resolve(setup)
    physics = Physics(setup)
    return SolverConfig(starting_temperatures=initial_temperatures, solver_physics=physics)


def add_form_default_values():
    """
    Change default form rendering on page load.

    Add default form values i.e. {'<variable name>': {'value': 10}}
    Ignore some of the input fields i.e. {'<variable name>': {'action': 'ignore'}}
    """
    return {
        "thermal_conductivity": {
            "action": "update",
            "value": 0.6
        },
        "heat_capacity": {
            "action": "update",
            "value": 4200
        },
        "density": {
            "action": "update",
            "value": 1000
        },
        "grid_spacing_meters": {
            "action": "update",
            "value": 0.001
        },
        "number_of_grid_nodes": {
            "action": "update",
            "value": 10
        },
        "simulation_time_seconds": {
            "action": "update",
            "value": 1000
        },
        "boundary_temperature_start": {
            "action": "update",
            "value": 100
        },
        "boundary_temperature_end": {
            "action": "update",
            "value": 20
        },
        "starting_temperature": {
            "action": "update",
            "value": 20
        },
    }


def load_boundary_conditions_start(form_data):
    boundary_condition = form_data["boundary_condition_start"]["value"]
    if boundary_condition == BOUNDARY_CONSTANT:
        return {
            "boundary_temperature_start": {
                "action": "update",
                "value": 100
            },
            "heat_trans_coeff_start": {
                "action": "delete"
            },
            "boundary_start_msg": {
                "action": "delete"
            }
        }
    elif boundary_condition == BOUNDARY_CONVECTIVE:
        return {
            "heat_trans_coeff_start": {
                "action": "update",
                "value": 5
            },
            "boundary_temperature_start": {
                "action": "update",
                "value": 100
            },
            "boundary_start_msg": {
                "action": "delete"
            }
        }
    else:
        return {
            "boundary_start_msg": {
                "action": "update",
                "value": "No info required for insulated boundary"
            },
            "boundary_temperature_start": {
                "action": "delete"
            },
            "heat_trans_coeff_start": {
                "action": "delete"
            }
        }


def load_boundary_conditions_end(form_data):
    boundary_condition = form_data["boundary_condition_end"]["value"]
    if boundary_condition == BOUNDARY_CONSTANT:
        return {
            "boundary_temperature_end": {
                "action": "update",
                "value": 100
            },
            "heat_trans_coeff_end": {
                "action": "delete"
            },
            "boundary_end_msg": {
                "action": "delete"
            }
        }
    elif boundary_condition == BOUNDARY_CONVECTIVE:
        return {
            "heat_trans_coeff_end": {
                "action": "update",
                "value": 5
            },
            "boundary_temperature_end": {
                "action": "update",
                "value": 100
            },
            "boundary_end_msg": {
                "action": "delete"
            }
        }
    else:
        return {
            "boundary_end_msg": {
                "action": "update",
                "value": "No info required for insulated boundary"
            },
            "boundary_temperature_end": {
                "action": "delete"
            },
            "heat_trans_coeff_end": {
                "action": "delete"
            }
        }


def _package_form_data(form_data) -> Config:
    return Config(
        thermal_conductivity=form_data["thermal_conductivity"],
        heat_capacity=form_data["heat_capacity"],
        density=form_data["density"],
        grid_spacing_meters=form_data["grid_spacing_meters"],
        number_of_grid_nodes=int(form_data["number_of_grid_nodes"]),
        simulation_time_seconds=form_data["simulation_time_seconds"],
        boundary_temperature_start=form_data.get("boundary_temperature_start"),
        boundary_temperature_end=form_data.get("boundary_temperature_end"),
        heat_trans_coeff_start=form_data.get("heat_trans_coeff_start"),
        heat_trans_coeff_end=form_data.get("heat_trans_coeff_end"),
        starting_temperature=form_data["starting_temperature"],
        boundary_condition_start=form_data["boundary_condition_start"]["value"],
        boundary_condition_end=form_data["boundary_condition_end"]["value"],
    )


def main(form_data: dict):
    """
    The function which gets executed upon the user pressing submit button.

    Args:
        form_data(dict): A dictionary of the field data. The keys to the form
                         data dictionary are the field names.
    """
    boundary_start = form_data["boundary_condition_start"]["value"]
    boundary_end = form_data["boundary_condition_end"]["value"]
    if boundary_start is None:
        raise Warning("Boundary conditions not loaded!")

    if (boundary_start in (BOUNDARY_CONSTANT, BOUNDARY_CONVECTIVE)
            and form_data.get('boundary_temperature_start') is None):
        raise Warning(
            f"Missing boundary temperature start for boundary condition {boundary_start}")

    if (boundary_end in (BOUNDARY_CONSTANT, BOUNDARY_CONVECTIVE)
            and form_data.get("boundary_temperature_end") is None):
        raise Warning(
            f"Missing boundary temperature end for boundary condition {boundary_end}")

    if (boundary_start == BOUNDARY_CONVECTIVE
            and form_data.get("heat_trans_coeff_start") is None):
        raise Warning(
            'Convective heat transfer coefficient must be supplied for this boundary condition.'
        )
    if (boundary_end == BOUNDARY_CONVECTIVE
            and form_data.get("heat_trans_coeff_end") is None):
        raise Warning(
            'Convective heat transfer coefficient must be supplied for this boundary condition.'
        )

    conf = _package_form_data(form_data)

    solver_config = get_solver_config(conf)

    times, temperatures = Solver.solve(solver_config, conf)
    fig = plt.figure(figsize=[10, 8])
    df = pd.DataFrame(temperatures,
                      columns=[f'Temp @node{i}' for i in range(len(temperatures[0]))])
    df.index = times
    ax = plt.gca()

    df.plot(style="-x", ax=ax)
    df['time'] = times

    plt.title(f"Temperatures in time for different boundary conditions:"
              f"\nStart boundary: '{boundary_start}'\n End boundary: '{boundary_end}'"
              f"\nsource: www.poppyi.com")
    plt.ylabel("Temperatures")
    plt.xlabel("Time (seconds)")
    plt.legend()
    plt.grid()
    imgdata = io.StringIO()
    fig.savefig(imgdata, format='svg')
    return {
        'plot_results': {
            'action': 'update',
            'value': imgdata.getvalue()
        },
        'output_table': {
            'action': 'update',
            'value': df.to_html(index=False)
        }
    }
