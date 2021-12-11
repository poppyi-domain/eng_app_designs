import pytest
import numpy as np
from designs import conduction_1d
from designs.conduction_1d import Config


@pytest.fixture(params=[{
    "value": "constant temperature start insulated end"
}, {
    "value": "constant temperature start and end"
}, {
    "value": "convective start insulated end"
}])
def boundary_conditions(request):
    return request.param


@pytest.fixture
def form_data_defaults() -> dict:
    return {
        "thermal_conductivity": 0.6,
        "heat_capacity": 4200,
        "density": 1000,
        "grid_spacing_meters": 0.001,
        "number_of_grid_nodes": 10,
        "simulation_time_seconds": 1000,
        "boundary_temperature_start": 100,
        "boundary_temperature_end": 20,
        "starting_temperature": 20,
        "convective_heat_trans_coeff": 5,
        "boundary_condition": {
            "value": "constant temperature start insulated end"
        }  # type: ignore
    }


@pytest.fixture()
def conf(form_data_defaults):
    return conduction_1d._package_form_data(form_data_defaults)


def test_returns_correct_response_with_different_boundary_conditions(
        form_data_defaults, boundary_conditions):
    form_data_defaults["boundary_condition"] = boundary_conditions
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_returns_correct_response(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "convective start insulated end"
    }  # type: ignore
    form_data_defaults["convective_heat_trans_coeff"] = 1
    result = conduction_1d.main(form_data_defaults)


def test_missing_convective_heat_trans_coeff_raises_warning(form_data_defaults):
    form_data_defaults["boundary_condition"] = {"value": "convective start insulated end"}
    form_data_defaults["convective_heat_trans_coeff"] = None
    with pytest.raises(Warning) as error_info:
        conduction_1d.main(form_data_defaults)

    assert str(
        error_info.value
    ) == "Convective heat transfer coefficient must be supplied for this boundary condition."


def test_missing_boundary_info_raises_warning(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "constant temperature start insulated end"
    }  # type: ignore
    del form_data_defaults["boundary_temperature_start"]

    with pytest.raises(Warning):
        conduction_1d.main(form_data_defaults)


def test_can_ignore_boundary_temperature_end_for_insulated_end_boundary(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "constant temperature start insulated end"
    }  # type: ignore
    del form_data_defaults["boundary_temperature_end"]
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_cannot_ignore_boundary_temperature_end_for_constant_temp_boundary(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "constant temperature start and end"
    }  # type: ignore
    del form_data_defaults["boundary_temperature_end"]
    with pytest.raises(Warning) as error_info:
        conduction_1d.main(form_data_defaults)

    assert str(error_info.value) == (
        "Missing boundary temperature end for boundary condition constant temperature start and end"
    )


def test_convective_energy_equal_to_accumulation(conf: Config):
    conf["boundary_conditions"] = "convective start insulated end"
    conf["convective_heat_trans_coeff"] = 1
    conf["simulation_time_seconds"] = 500
    conf["starting_temperature"] = 20
    solver_config = conduction_1d.get_solver_config(conf)
    times, temperatures = conduction_1d.Solver.solve(solver_config, conf)
    average_temp_end = np.mean(temperatures[-1])
    start_temp = temperatures[0][0]

    assert start_temp == 20
    assert average_temp_end - start_temp == pytest.approx(1.0773369387960905)

    # m * cp * deltaT
    assert (average_temp_end -
            start_temp) * 0.009 * 4200 * 1000 == pytest.approx(40723.33628649221)

    # h * A * (T_inf - T_avg) * delta_time
    # i.e 1 * 500 * (100 - 20) == approx(40000)
