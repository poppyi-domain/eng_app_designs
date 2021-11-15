import pytest
from designs import concentric_tube_hx
from tests import fluid_data


def temperature_vary_annulus_cold():
    return {"configuration": "fluid annulus cold"}


def temperature_vary_annulus_hot():
    return {"configuration": "fluid annulus hot"}


def temperature_vary_fluid_pipe_cold():
    return {"configuration": "fluid pipe cold"}


def temperature_vary_fluid_pipe_hot():
    return {"configuration": "fluid pipe hot"}


def hx_flow_type_counter_flow():
    return {"flow_type": "counter-flow"}


def hx_flow_type_parrallel_flow():
    return {"flow_type": "parallel-flow"}


def example_data():
    return {
        "fluid_pipe": fluid_data.FLUID_DATA_AIR,
        "fluid_pipe_hot_temp_c": 100,
        "fluid_pipe_cold_temp_c": 85,
        "fluid_pipe_flow_rate_kg_s": 0.1,
        "fluid_annulus": fluid_data.FLUID_DATA_ACETYLENE,
        "fluid_annulus_hot_temp_c": 80,
        "fluid_annulus_cold_temp_c": 70,
        "fluid_annulus_flow_rate_kg_s": 0.1,
        "temperature_variable": temperature_vary_fluid_pipe_cold(),
        "dia_inner_annulus_mm": 100,
        "dia_outer_pipe_mm": 90,
        "dia_inner_pipe_mm": 85,
        "thermal_cond_pipe_w_m_k": 10,
        "hx_flow_type": hx_flow_type_parrallel_flow(),
    }


def test_correct_temperature_is_modified_for_annulus_cold():
    form_data = example_data()
    form_data["temperature_variable"] = temperature_vary_annulus_cold()
    output = concentric_tube_hx.main(form_data)

    assert output["fluid_pipe_hot_temp_c_out"]["value"] == 100
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == 85
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == 80
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == pytest.approx(71.664520867)


def test_correct_temperature_is_modified_for_annulus_hot():
    form_data = example_data()
    form_data["temperature_variable"] = temperature_vary_annulus_hot()
    output = concentric_tube_hx.main(form_data)
    assert output["fluid_pipe_hot_temp_c_out"]["value"] == 100
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == 85
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == pytest.approx(78.35112799926003)
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == 70


def test_correct_temperature_is_modified_for_pipe_cold():
    form_data = example_data()
    form_data["temperature_variable"] = temperature_vary_fluid_pipe_cold()
    output = concentric_tube_hx.main(form_data)
    assert output["fluid_pipe_hot_temp_c_out"]["value"] == 100
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == pytest.approx(82.0190172855111)
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == 80
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == 70


def test_correct_temperature_is_modified_for_pipe_hot():
    form_data = example_data()
    form_data["temperature_variable"] = temperature_vary_fluid_pipe_hot()
    output = concentric_tube_hx.main(form_data)
    assert output["fluid_pipe_hot_temp_c_out"]["value"] == pytest.approx(102.97587481917554)
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == 85
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == 80
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == 70


def test_does_raise_warning_for_temperature_violations():
    form_data = example_data()
    form_data["fluid_pipe_hot_temp_c"] = 125
    form_data["fluid_pipe_flow_rate_kg_s"] = 0.3
    form_data["temperature_variable"] = temperature_vary_annulus_hot()
    with pytest.raises(Warning) as error:
        output = concentric_tube_hx.main(form_data)
    assert 'Error! Heat exchanged from cold to hot! Check fluid inlet temperatures and flow rates.' in str(
        error.value)


def engine_oil_example_data():
    return {
        "fluid_pipe": fluid_data.WATER,
        "fluid_pipe_hot_temp_c": 40,
        "fluid_pipe_cold_temp_c": 30,
        "fluid_pipe_flow_rate_kg_s": 0.2,
        "fluid_annulus": fluid_data.ENGINE_OIL,
        "fluid_annulus_hot_temp_c": 100,
        "fluid_annulus_cold_temp_c": 60,
        "fluid_annulus_flow_rate_kg_s": 0.1,
        "temperature_variable": temperature_vary_fluid_pipe_hot(),
        "dia_inner_annulus_mm": 45,
        "dia_outer_pipe_mm": 25,
        "dia_inner_pipe_mm": 24.999,
        "thermal_cond_pipe_w_m_k": 100,
        "hx_flow_type": hx_flow_type_counter_flow(),
    }


def test_outputs_engine_oil_problem():
    form_data = engine_oil_example_data()
    output = concentric_tube_hx.main(form_data)
    assert output["fluid_pipe_hot_temp_c_out"]["value"] == pytest.approx(40.201053)
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == 30
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == 100
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == 60
    assert output["log_mean_temp"]["value"] == pytest.approx(43.199985501726445)
    assert output["reynolds_pipe"]["value"] == pytest.approx(14050.101808)
    assert output["nusselt_pipe"]["value"] == pytest.approx(89.958453)
    assert output["heat_trans_coeff_pipe"]["value"] == pytest.approx(2249.05131072)
    assert output["reynolds_annulus"]["value"] == pytest.approx(55.9665733)
    # TODO: Include warning comments about transitional regimes.
    assert output["nusselt_annulus"]["value"] == pytest.approx(5.64222)
    assert output["heat_trans_coeff_annulus"]["value"] == pytest.approx(38.93133)
    assert output["overall_heat_transfer_coeff"]["value"] == pytest.approx(38.26886)
    assert output["heat_exchanger_length"]["value"] == pytest.approx(65.6484416)


def test_outputs_engine_oil_parrallel_flow():
    form_data = engine_oil_example_data()
    form_data["hx_flow_type"] = hx_flow_type_parrallel_flow()
    output = concentric_tube_hx.main(form_data)
    assert output["fluid_pipe_hot_temp_c_out"]["value"] == pytest.approx(40.201053)
    assert output["fluid_pipe_cold_temp_c_out"]["value"] == 30
    assert output["fluid_annulus_hot_temp_c_out"]["value"] == 100
    assert output["fluid_annulus_cold_temp_c_out"]["value"] == 60
    assert output["log_mean_temp"]["value"] == pytest.approx(39.75167078795913)
    assert output["reynolds_pipe"]["value"] == pytest.approx(14050.101808)
    assert output["nusselt_pipe"]["value"] == pytest.approx(89.958453)
    assert output["heat_trans_coeff_pipe"]["value"] == pytest.approx(2249.05131072)
    assert output["reynolds_annulus"]["value"] == pytest.approx(55.9665733)
    assert output["heat_trans_coeff_annulus"]["value"] == pytest.approx(38.93133)
    assert output["overall_heat_transfer_coeff"]["value"] == pytest.approx(38.26886)
    assert output["heat_exchanger_length"]["value"] == pytest.approx(71.343208)


def test_does_raise_in_transitional_regime_pipe():
    form_data = engine_oil_example_data()
    form_data["hx_flow_type"] = hx_flow_type_parrallel_flow()
    form_data["fluid_pipe_flow_rate_kg_s"] = 0.1
    with pytest.raises(Warning) as error:
        output = concentric_tube_hx.main(form_data)
    assert str(error.value) == (
        "Average Reynolds number of fluid in pipe calculated as 7025.05 which is in "
        "transitional regime results unreliable. Acceptable range is: 2300 > Reynolds > 10000")


def test_does_raise_warning_if_pipe_diameters_incorrect():
    form_data = engine_oil_example_data()
    form_data["dia_inner_annulus_mm"] = 2.5
    with pytest.raises(Warning) as error:
        output = concentric_tube_hx.main(form_data)
    assert str(error.value) == 'Annulus pipe diameter must be greater than the pipe!'

    form_data = engine_oil_example_data()
    form_data["dia_outer_pipe_mm"] = 22
    with pytest.raises(Warning) as error:
        output = concentric_tube_hx.main(form_data)
    assert str(error.value) == 'Outer pipe diamter must be greater than inner diameter.'


def test_does_raise_in_transitional_regime_annulus():
    form_data = engine_oil_example_data()
    form_data["hx_flow_type"] = hx_flow_type_parrallel_flow()
    form_data["fluid_annulus_hot_temp_c"] = 60
    form_data["temperature_variable"] = temperature_vary_fluid_pipe_cold()
    form_data["dia_inner_annulus_mm"] = 35
    form_data["fluid_annulus_flow_rate_kg_s"] = 5
    with pytest.raises(Warning) as error:
        output = concentric_tube_hx.main(form_data)
        assert output["reynolds_annulus"]["value"] == 1

    assert str(error.value) == (
        'Average Annulus fluid Reynolds number calculated as 3264.72 which is in '
        'transitional regime results unreliable. Acceptable range is: 2300 > Reynolds '
        '> 10000')
