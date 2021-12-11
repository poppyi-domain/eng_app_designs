import io
import numpy as np
import pandas as pd
from typing import Callable
from matplotlib import pyplot as plt
from scipy import integrate
from typing import TypedDict, Optional

BOUND_CONSTANT_TEMP_START_END = "constant temperature start and end"
BOUND_CONTANT_TEMP_START_INSULATED_END = "constant temperature start insulated end"
BOUND_CONV_START_INSULATED_END = "convective start insulated end"


class Config(TypedDict):
    thermal_conductivity: float
    heat_capacity: float
    density: float
    grid_spacing_meters: float
    number_of_grid_nodes: int
    simulation_time_seconds: float
    boundary_conditions: str
    boundary_temperature_start: float
    boundary_temperature_end: Optional[float]
    convective_heat_trans_coeff: Optional[float]
    starting_temperature: float


class SolverConfig(TypedDict):
    solver_physics: Callable
    starting_temperatures: list


class ResolveInitialTemperatures:
    """Configure the boundary conditions."""
    def __init__(self):
        self._map = {
            BOUND_CONSTANT_TEMP_START_END: self._configure_constant_temperature,
            BOUND_CONTANT_TEMP_START_INSULATED_END: self._configure_insulated_boundary,
            BOUND_CONV_START_INSULATED_END: self._configure_conv_start_insulated_end,
        }

    def resolve(self, setup: Config):
        if setup["number_of_grid_nodes"] > 20:
            raise Warning("Number of grid nodes too high")

        func = self._map[setup["boundary_conditions"]]
        return func(setup)

    @staticmethod
    def _configure_constant_temperature(setup: Config):
        temperatures_initial = np.array([setup["starting_temperature"]] *
                                        setup["number_of_grid_nodes"])
        temperatures_initial[0] = setup["boundary_temperature_start"]
        temperatures_initial[-1] = setup["boundary_temperature_end"]
        return temperatures_initial

    @staticmethod
    def _configure_insulated_boundary(setup: Config):
        temperatures_initial = np.array([setup["starting_temperature"]] *
                                        setup["number_of_grid_nodes"])
        temperatures_initial[0] = setup["boundary_temperature_start"]
        return temperatures_initial

    @staticmethod
    def _configure_conv_start_insulated_end(setup: Config):
        temperatures_initial = np.array([setup["starting_temperature"]] *
                                        setup["number_of_grid_nodes"])
        return temperatures_initial


class PhysicsFactory:
    def __init__(self):
        self._map = {
            BOUND_CONSTANT_TEMP_START_END: self._constant_temperature,
            BOUND_CONTANT_TEMP_START_INSULATED_END:
            self._constant_temperature_start_insulated_end,
            BOUND_CONV_START_INSULATED_END: self._convective_start_insulated_end,
        }

    def resolve(self, setup: Config):
        return self._map[setup["boundary_conditions"]]

    @staticmethod
    def _constant_temperature(t, temperatures, config: Config):
        """Return the temperature gradients based on the current temperature profile."""

        # Solver for where boundary conditions are constant temperature at start and end.
        constant = config["thermal_conductivity"] / (config["heat_capacity"] *
                                                     config["density"])
        gradients = np.zeros(len(temperatures))
        gradients[1:-1] = (constant *
                           (temperatures[:-2] - 2 * temperatures[1:-1] + temperatures[2:]) /
                           config["grid_spacing_meters"]**2)
        return gradients

    @staticmethod
    def _constant_temperature_start_insulated_end(t, temperatures, config: Config):
        gradients = PhysicsFactory._constant_temperature(t, temperatures, config)

        constant = config["thermal_conductivity"] / (config["heat_capacity"] *
                                                     config["density"])

        gradients[-1] = (constant * (temperatures[-2] - temperatures[-1]) /
                         config["grid_spacing_meters"]**2)
        return gradients

    @staticmethod
    def _convective_start_insulated_end(t, temperatures, config: Config):
        gradients = PhysicsFactory._constant_temperature_start_insulated_end(
            t, temperatures, config)

        h = config["convective_heat_trans_coeff"]
        k = config["thermal_conductivity"]
        T_infinity = config["boundary_temperature_start"]
        c = 1 / (config["density"] * config["heat_capacity"] * config["grid_spacing_meters"])

        gradients[0] = (c *
                        (h * (T_infinity - temperatures[0]) + k *
                         (temperatures[2] - temperatures[1]) / config["grid_spacing_meters"]))
        return gradients


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
    physics = PhysicsFactory().resolve(setup)
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


def load_boundary_conditions(form_data):
    if form_data["boundary_condition"]["value"] == "constant temperature start and end":
        return {
            "boundary_temperature_start": {
                "action": "update",
                "value": 100
            },
            "boundary_temperature_end": {
                "action": "update",
                "value": 20
            }
        }
    elif form_data["boundary_condition"][
            "value"] == "constant temperature start insulated end":
        return {
            "boundary_temperature_start": {
                "action": "update",
                "value": 100
            },
            "boundary_temperature_end": {
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
        boundary_temperature_start=form_data["boundary_temperature_start"],
        boundary_temperature_end=form_data.get("boundary_temperature_end"),
        convective_heat_trans_coeff=form_data.get("convective_heat_trans_coeff"),
        starting_temperature=form_data["starting_temperature"],
        boundary_conditions=form_data["boundary_condition"]["value"],
    )


def main(form_data: dict):
    """
    The function which gets executed upon the user pressing submit button.

    Args:
        form_data(dict): A dictionary of the field data. The keys to the form
                         data dictionary are the field names.
    """
    if form_data.get("boundary_temperature_start") is None:
        raise Warning("Boundary conditions not loaded!")

    if (form_data["boundary_condition"]["value"] == "constant temperature start and end"
            and form_data.get("boundary_temperature_end") is None):
        raise Warning("Missing boundary temperature end for boundary condition {}".format(
            form_data["boundary_condition"]["value"]))

    if (form_data["boundary_condition"]["value"] == BOUND_CONV_START_INSULATED_END
            and form_data.get("convective_heat_trans_coeff") is None):
        raise Warning(
            'Convective heat transfer coefficient must be supplied for this boundary condition.'
        )

    conf = _package_form_data(form_data)

    solver_config = get_solver_config(conf)

    times, temperatures = Solver.solve(solver_config, conf)
    fig = plt.figure(figsize=[10, 8])
    df = pd.DataFrame(temperatures,
                      columns=[f'Temp @node{i}' for i in range(len(temperatures[0]))])
    df['time'] = times

    plt.plot(times, temperatures, "-x")
    plt.title("Temperatures in time")
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
