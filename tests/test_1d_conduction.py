import pytest
from designs import conduction_1d


@pytest.fixture(params=[{
    "value": "constant temperature start insulated end"
}, {
    "value": "constant temperature start and end"
}])
def boundary_conditions(request):
    return request.param


@pytest.fixture
def form_data_defaults():
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
        "boundary_condition": {
            "value": "constant temperature start insulated end"
        }
    }


def test_returns_correct_response_with_different_boundary_conditions(
        form_data_defaults, boundary_conditions):
    form_data_defaults["boundary_condition"] = boundary_conditions
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_missing_boundary_info_raises_warning(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "constant temperature start insulated end"
    }
    del form_data_defaults["boundary_temperature_start"]

    with pytest.raises(Warning):
        conduction_1d.main(form_data_defaults)


def test_can_ignore_boundary_temperature_end_for_insulated_end_boundary(form_data_defaults):
    form_data_defaults["boundary_condition"] = {
        "value": "constant temperature start insulated end"
    }
    del form_data_defaults["boundary_temperature_end"]
    result = conduction_1d.main(form_data_defaults)

    assert list(result) == ['plot_results', 'output_table']
    for value in result.values():
        assert value['action']
        assert value['value']


def test_cannot_ignore_boundary_temperature_end_for_constant_temp_boundary(form_data_defaults):
    form_data_defaults["boundary_condition"] = {"value": "constant temperature start and end"}
    del form_data_defaults["boundary_temperature_end"]
    with pytest.raises(Warning) as error_info:
        result = conduction_1d.main(form_data_defaults)

    assert str(error_info.value) == (
        "Missing boundary temperature end for boundary condition constant temperature start and end"
    )
