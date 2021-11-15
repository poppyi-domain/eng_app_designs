import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from scipy import integrate
from typing import TypedDict

BOUND_CONSTANT_TEMP_START_END = "constant temperature start and end"
BOUND_CONTANT_TEMP_START_INSULATED_END = "constant temperature start insulated end"
BOUND_CONSTANT_HEAT_FLUX = "constant heat flux start and end"


class Config(TypedDict):
    thermal_conductivity: float
    heat_capacity: float
    density: float
    grid_spacing_meters: float
    number_of_grid_nodes: int
    simulation_time_seconds: float
    boundary_conditions: str
    boundary_temperature_start: float
    boundary_temperature_end: float
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
            BOUND_CONSTANT_HEAT_FLUX: self._configure_constant_heat_flux_boundary,
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
    def _configure_constant_heat_flux_boundary(setup: Config):
        return np.array([setup["starting_temperature"]] * setup["number_of_grid_nodes"])


class PhysicsFactory:
    def __init__(self):
        self._map = {
            BOUND_CONSTANT_TEMP_START_END: self._constant_temperature,
            BOUND_CONTANT_TEMP_START_INSULATED_END:
            self._constant_temperature_start_insulated_end,
            BOUND_CONSTANT_HEAT_FLUX: None,
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


if __name__ == "__main__":
    conf = Config(
        thermal_conductivity=0.6,
        heat_capacity=4200,
        density=1000,
        grid_spacing_meters=1e-3,
        number_of_grid_nodes=10,
        simulation_time_seconds=1000,
        boundary_temperature_start=100,
        boundary_temperature_end=20,
        starting_temperature=20,
        boundary_conditions=BOUND_CONTANT_TEMP_START_INSULATED_END,
    )

    solver_config = get_solver_config(conf)

    times, temperatures = Solver.solve(solver_config, conf)
    plt.plot(times, temperatures, "-x")
    plt.title("Temperatures in time")
    plt.ylabel("Temperatures")
    plt.xlabel("Time (seconds)")
    plt.legend()
    plt.grid(b=True)
    plt.show()
