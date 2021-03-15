from designs import hot_wire_temperature


def nichrome_resistance_wire():
    # resistance is technically resistivity in ohm.mm^2/m
    return {"emissivity": 0.7,
            "resistance": 1.1}


def form_data_example():
    return {
        "wire_resistance": nichrome_resistance_wire(),
        "wire_diameter_mm": 2,
        "wire_length_meters": 1.5,
        "voltage": 20,
        "ambient_temp_deg_c": 25,
    }


def test_main_call_temperature():
    output = hot_wire_temperature.main(form_data_example())

    assert output["wire_resistance_ohms"]["value"] == 0.52521
    assert output["power_watts"]["value"] == 761.59822
    assert output["element_temp"]["value"] == 1169.12107
