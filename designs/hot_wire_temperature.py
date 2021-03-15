import numpy as np


def add_form_default_values():
    """
    Change default form rendering on page load.

    Add default form values i.e. {'<variable name>': {'value': 10}}
    Ignore some of the input fields i.e. {'<variable name>': {'action': 'ignore'}}
    """
    return {'ambient_temp_deg_c': {'action': 'update', 'value': 20}}


def area(d):
    """Cross sectional area."""
    return np.pi * d ** 2 / 4


def surface_area(d, L):
    """Surface area in m^2"""
    return np.pi * d * 1e-3 * L


def resistance(row, L, A):
    """A= area in mm^2     L = length of wire in meters     row = resistivity in ohm.mm^2/m"""
    return row * L / A


def power(V, R):
    """Electrical power Watts.     V in volts, R in ohms"""
    return V ** 2 / R


def element_temp(Q, L, d, emis, ambient_deg_c):
    h = 8  # W/m2.K  heat transfer coefficient- guess
    T0 = ambient_deg_c + 273.15  # Kelvin
    A2 = surface_area(d, L)  # surface area of wire
    T_arr = np.arange(0, 10000, 10)
    Q_arr = h * A2 * (T_arr - T0) + emis * 5.67e-8 * A2 * (T_arr ** 4 - T0 ** 4)
    return np.interp(Q, Q_arr, T_arr)


def validate_numeric(form_data, key):
    try:
        float(form_data[key])
    except ValueError:
        raise Warning('Numeric input required for: {}'.format(key))


def main(form_data):
    """
    The function which gets executed upon the user pressing submit button.
    Args:         form_data(dict): A dictionary of the field data. The keys to the form
                                   data dictionary are the field names.
    """

    row = form_data['wire_resistance']['resistance']
    d = form_data['wire_diameter_mm']
    emis = form_data['wire_resistance']['emissivity']
    V = form_data['voltage']
    L = form_data['wire_length_meters']
    ambient = form_data['ambient_temp_deg_c']

    for key in ['voltage', 'wire_diameter_mm', 'wire_length_meters', 'ambient_temp_deg_c']:
        validate_numeric(form_data, key)

    A = area(d)  # mm^2
    R = resistance(row, L, A)  # ohms
    Q = round(power(V, R), 5)  # Watts
    T = round(element_temp(Q, L, d, emis, ambient), 5)  # Kelvin
    return {'output_heading': {'action': 'update'},
            'container_outputs': {'action': 'update'},
            'wire_resistance_ohms': {"action": "update", "value": round(R, 5)},
            "power_watts": {"action": "update", "value": Q},
            "element_temp": {"action": "update", "value": T}}
