"""
Peng-robinson equation of state calculator.

As used: https://www.poppyi.com/app_design_form/public_render/peng%20robinson%20eq%20of%20state
"""
import numpy as np

R = 8.31446  # Pa.m3/K


def validate_numeric(form_data, key):
    try:
        return float(form_data[key])
    except ValueError:
        raise Warning('Numeric input required for: {}'.format(key))


def main(form_data):
    temperature_kelvin = validate_numeric(form_data, 'temp_kelvin')
    pressure_pascal = validate_numeric(form_data, 'pressure_pascal')
    temperature_critical = validate_numeric(form_data, 'critical_temp_kelvin')
    pressure_critical = validate_numeric(form_data, 'critical_pressure_pa')
    acentric_factor = validate_numeric(form_data, 'acentric_factor')

    temperature_critical = float(temperature_critical)
    pressure_critical = float(pressure_critical)
    acentric_factor = float(acentric_factor)

    a = (0.457235 * (R * temperature_critical) ** 2) / pressure_critical;
    b = 0.077796 * R * temperature_critical / pressure_critical;

    if acentric_factor <= 0.49:
        kappa = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor ** 2
    else:
        kappa = 0.379642 + 1.48503 * acentric_factor - 0.164423 * acentric_factor ** 2 + 0.0166666 * acentric_factor ** 3

    reduced_temp = temperature_kelvin / temperature_critical

    alpha = (1 + kappa * (1 - reduced_temp ** 0.5)) ** 2

    A = alpha * a * pressure_pascal / (R * temperature_kelvin) ** 2
    B = b * pressure_pascal / (R * temperature_kelvin)

    k3 = 1
    k2 = 1 - B
    k1 = A - 2 * B - 3 * B ** 2
    k0 = A * B - B ** 2 - B ** 3

    z_roots = np.roots([k3, -k2, k1, -k0])

    z = z_roots.real[z_roots.imag < 1e-5]

    z_filtered = [float(z_) for z_ in z if z_ >= 1e-3]

    if len(z_filtered) == 0:
        raise Warning('peng robinson eq of state error: no solutions found (no roots)')

    z_str = [str(i) for i in z_filtered]
    return {'compressibility_max': {'action': 'update', 'value': max(z_filtered)},
            'compressibility_min': {'action': 'update', 'value': min(z_filtered)},
            'peng_calc_output': {'action': 'update'},
            'debug_outputs': {'action': 'update'},
            'critical_temp': {'action': 'update', 'value': temperature_critical},
            'critical_pressure': {'action': 'update', 'value': pressure_critical},
            'acentric_factor': {'action': 'update', 'value': acentric_factor},
            'reduced_temp': {'action': 'update', 'value': reduced_temp},
            'a': {'action': 'update', 'value': a},
            'b': {'action': 'update', 'value': b},
            'k': {'action': 'update', 'value': kappa},
            'alpha': {'action': 'update', 'value': alpha},
            'A': {'action': 'update', 'value': A},
            'B': {'action': 'update', 'value': B},
            }
