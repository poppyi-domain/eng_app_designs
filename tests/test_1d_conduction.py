import pandas as pd
import pytest
import numpy as np
from designs import conduction_1d
from designs.conduction_1d import Config


@pytest.fixture(params=[{
    "value": conduction_1d.BOUNDARY_CONVECTIVE
}, {
    "value": conduction_1d.BOUNDARY_INSULATED
}, {
    "value": conduction_1d.BOUNDARY_CONSTANT
}],
                ids=[
                    conduction_1d.BOUNDARY_CONVECTIVE, conduction_1d.BOUNDARY_INSULATED,
                    conduction_1d.BOUNDARY_CONSTANT
                ])
def boundary_conditions_start(request):
    return request.param


@pytest.fixture(params=[{
    "value": conduction_1d.BOUNDARY_CONVECTIVE
}, {
    "value": conduction_1d.BOUNDARY_INSULATED
}, {
    "value": conduction_1d.BOUNDARY_CONSTANT
}],
                ids=[
                    conduction_1d.BOUNDARY_CONVECTIVE, conduction_1d.BOUNDARY_INSULATED,
                    conduction_1d.BOUNDARY_CONSTANT
                ])
def boundary_conditions_end(request):
    return request.param


@pytest.fixture(params=[{
    "value": conduction_1d.BOUNDARY_CONVECTIVE
}, {
    "value": conduction_1d.BOUNDARY_CONSTANT
}],
                ids=[conduction_1d.BOUNDARY_CONVECTIVE, conduction_1d.BOUNDARY_CONSTANT])
def boundaries_start_requiring_temperature(request):
    return request.param


@pytest.fixture(params=[{
    "value": conduction_1d.BOUNDARY_CONVECTIVE
}, {
    "value": conduction_1d.BOUNDARY_CONSTANT
}],
                ids=[conduction_1d.BOUNDARY_CONVECTIVE, conduction_1d.BOUNDARY_CONSTANT])
def boundaries_end_requiring_temperature(request):
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
        "boundary_temperature_end": 100,
        "starting_temperature": 20,
        "heat_trans_coeff_start": 5,
        "heat_trans_coeff_end": 5,
        "boundary_condition_start": {
            "value": conduction_1d.BOUNDARY_CONSTANT
        },
        "boundary_condition_end": {
            "value": conduction_1d.BOUNDARY_CONSTANT
        }
    }


@pytest.fixture()
def conf(form_data_defaults):
    return conduction_1d._package_form_data(form_data_defaults)


def test_returns_correct_response_with_different_boundary_conditions(
        form_data_defaults, boundary_conditions_start, boundary_conditions_end):
    form_data_defaults["boundary_condition_start"] = boundary_conditions_start
    form_data_defaults["boundary_condition_end"] = boundary_conditions_end
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


@pytest.mark.parametrize("boundary_key",
                         ["boundary_condition_start", "boundary_condition_end"])
def test_missing_convective_heat_trans_coeff_raises_warning(boundary_key, form_data_defaults):
    form_data_defaults[boundary_key] = {"value": conduction_1d.BOUNDARY_CONVECTIVE}
    form_data_defaults["heat_trans_coeff_start"] = None
    form_data_defaults["heat_trans_coeff_end"] = None
    with pytest.raises(Warning) as error_info:
        conduction_1d.main(form_data_defaults)

    assert str(
        error_info.value
    ) == "Convective heat transfer coefficient must be supplied for this boundary condition."


def test_heat_transfer_coeff_end_can_be_none(form_data_defaults):
    form_data_defaults["boundary_condition_start"] = {"value": conduction_1d.BOUNDARY_CONVECTIVE}
    form_data_defaults["boundary_condition_end"] = {"value": conduction_1d.BOUNDARY_INSULATED}
    form_data_defaults["heat_trans_coeff_start"] = 5
    form_data_defaults["heat_trans_coeff_end"] = None
    conduction_1d.main(form_data_defaults)


def test_missing_boundary_temperature_start_raises_warning(
        form_data_defaults, boundaries_start_requiring_temperature, boundary_conditions_end):
    form_data_defaults["boundary_condition_start"] = boundaries_start_requiring_temperature
    form_data_defaults["boundary_condition_end"] = boundary_conditions_end
    del form_data_defaults["boundary_temperature_start"]

    with pytest.raises(Warning) as error_info:
        conduction_1d.main(form_data_defaults)

    assert str(error_info.value).startswith(
        "Missing boundary temperature start for boundary condition")


def test_missing_boundary_temperature_end_raises_warning(form_data_defaults,
                                                         boundary_conditions_start,
                                                         boundaries_end_requiring_temperature):
    form_data_defaults["boundary_condition_start"] = boundary_conditions_start
    form_data_defaults["boundary_condition_end"] = boundaries_end_requiring_temperature
    del form_data_defaults["boundary_temperature_end"]

    with pytest.raises(Warning) as error_info:
        conduction_1d.main(form_data_defaults)
    assert str(
        error_info.value).startswith("Missing boundary temperature end for boundary condition")


def test_can_ignore_boundary_temperature_end_for_insulated_end_boundary(
        form_data_defaults, boundary_conditions_start):
    form_data_defaults["boundary_condition_end"] = {"value": conduction_1d.BOUNDARY_INSULATED}
    form_data_defaults["boundary_conditions_start"] = boundary_conditions_start
    del form_data_defaults["boundary_temperature_end"]
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_can_ignore_boundary_temperature_start_for_insulated_start_boundary(
        form_data_defaults, boundary_conditions_end):
    form_data_defaults["boundary_condition_start"] = {
        "value": conduction_1d.BOUNDARY_INSULATED
    }
    form_data_defaults["boundary_condition_end"] = boundary_conditions_end
    del form_data_defaults["boundary_temperature_start"]
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_convective_energy_equal_to_accumulation(conf: Config):
    # Performing energy balance check where temperature change is approximately constant.
    conf["boundary_condition_start"] = conduction_1d.BOUNDARY_CONVECTIVE
    conf["boundary_condition_end"] = conduction_1d.BOUNDARY_INSULATED
    conf["heat_trans_coeff_start"] = 1
    conf["simulation_time_seconds"] = 500
    conf["starting_temperature"] = 20
    solver_config = conduction_1d.get_solver_config(conf)
    times, temperatures = conduction_1d.Solver.solve(solver_config, conf)
    average_temp_end = np.mean(temperatures[-1])
    start_temp = temperatures[0][0]

    df = pd.DataFrame(temperatures)
    df['times'] = times

    assert start_temp == 20
    assert average_temp_end - start_temp == pytest.approx(1.0773369387960905)

    # m * cp * deltaT
    assert (average_temp_end -
            start_temp) * 0.009 * 4200 * 1000 == pytest.approx(40723.33628649221)

    # h * A * (T_inf - T_avg) * delta_time
    # i.e 1 * 500 * (100 - 20) == approx(40000)
