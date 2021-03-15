import numpy as np
from matplotlib import pyplot as plt


def add_form_default_values():
    return {}


class Hx:
    def __init__(self,
                 annulus_fluid_data: dict,
                 pipe_fluid_data: dict,
                 k_wall: float,
                 t_annulus_cold_kelvin: float,
                 t_annulus_hot_kelvin: float,
                 t_pipe_hot_kelvin: float,
                 t_pipe_cold_kelvin: float,
                 dia_annulus_inner_m: float,
                 dia_pipe_outer_m: float,
                 dia_pipe_inner_m: float,
                 mass_flow_annulus: float,
                 mass_flow_pipe: float,
                 temperature_configuration_type: str,
                 flow_configurate_type: str):

        self.k_wall = k_wall
        self._annulus_fluid = annulus_fluid_data
        self._pipe_fluid = pipe_fluid_data

        self._mass_flow_annulus = mass_flow_annulus  # kg/s
        self._mass_flow_pipe = mass_flow_pipe  # kg/s
        self.t_annulus_cold = t_annulus_cold_kelvin
        self.t_annulus_hot = t_annulus_hot_kelvin
        self.t_pipe_hot = t_pipe_hot_kelvin
        self.t_pipe_cold = t_pipe_cold_kelvin

        self._dia_annulus_inner_m = dia_annulus_inner_m
        self._dia_pipe_outer_m = dia_pipe_outer_m
        self._dia_pipe_inner_m = dia_pipe_inner_m

        self._configuration_type = temperature_configuration_type
        self._flow_configuration_type = flow_configurate_type

        self.q_exchanged = None

        self._setup()

    def _setup(self):
        self.check_pipe_temperatures_initial()
        self._check_pipe_diameters_sensible()

        for _ in range(10):
            # iterative as the temperatures affect the heat capacity
            # hence it's done a number of times
            self.calculate_inlet_outlet_temperatures()
        self.check_pipe_temperatures_final()

    def _check_pipe_diameters_sensible(self):
        if self._dia_annulus_inner_m < self._dia_pipe_outer_m:
            raise Warning('Annulus pipe diameter must be greater than the pipe!')

        if self._dia_pipe_inner_m > self._dia_pipe_outer_m:
            raise Warning('Outer pipe diamter must be greater than inner diameter.')

    @property
    def annulus_mean_temp(self):
        return (self.t_annulus_cold + self.t_annulus_hot) / 2

    @property
    def pipe_mean_temp(self):
        return (self.t_pipe_cold + self.t_pipe_hot) / 2

    @property
    def cp_annulus(self):
        return self._get_cp_fluid(self.annulus_mean_temp, self._annulus_fluid)

    @property
    def cp_pipe(self):
        return self._get_cp_fluid(self.pipe_mean_temp, self._pipe_fluid)

    def _get_cp_fluid(self, fluid_temp_mean_kelvin, fluid_type):
        cp = fluid_type['heat_capacity_cp']['cp_j_kg_k']
        temp = fluid_type['heat_capacity_cp']['temp_range_k']
        if np.max(fluid_temp_mean_kelvin) > max(temp):
            raise Warning('Temperature exceeds Cp data maximum temperature value of {} Kelvin'.format(max(temp)))
        elif np.min(fluid_temp_mean_kelvin) < min(temp):
            raise Warning('Temperature below Cp data minimum value of {} Kelvin'.format(min(temp)))
        else:
            return np.interp(fluid_temp_mean_kelvin, temp, cp)

    @property
    def visc_annulus(self):
        return self._get_visc(self.annulus_mean_temp, self._annulus_fluid)

    @property
    def visc_pipe(self):
        return self._get_visc(self.pipe_mean_temp, self._pipe_fluid)

    def _get_visc(self, fluid_temp_mean_kelvin, fluid_type):
        visc = fluid_type['viscosity']['visc_pa_s']
        temp = fluid_type['viscosity']['temp_range_k']
        if np.max(fluid_temp_mean_kelvin) > max(temp):
            raise Warning('Temperature exceeds maximum value of {}'.format(max(temp)))
        elif np.min(fluid_temp_mean_kelvin) < min(temp):
            raise Warning('Temperature below minimum value of {}'.format(min(temp)))
        else:
            return np.interp(fluid_temp_mean_kelvin, temp, visc)

    @property
    def k_annulus(self):
        return self._get_k_fluid(self.annulus_mean_temp, self._annulus_fluid)

    @property
    def k_pipe(self):
        return self._get_k_fluid(self.pipe_mean_temp, self._pipe_fluid)

    @property
    def prandtl_annulus(self):
        return self.cp_annulus * self.visc_annulus / self.k_annulus

    @property
    def prandtl_pipe(self):
        return self.cp_pipe * self.visc_pipe / self.k_pipe

    @property
    def nusselt_pipe(self):
        re = self.reynolds_pipe()
        pr = self.prandtl_pipe
        if self.reynolds_pipe() < 2300:
            return 3.66
        elif re > 10000:
            if pr < 0.7 or pr > 160:
                raise Warning('Inner pipe Prandtl number is {:.4f}, which is out of range: 0.7 < Pr < 160'.format(
                    self.prandtl_pipe))
            return 0.023 * self.reynolds_pipe() ** 0.8 * self.prandtl_pipe ** 0.4
        else:
            raise Warning('Average Reynolds number of fluid in pipe calculated as {:.2f} which is in transitional regime'
                          ' results unreliable. Acceptable range is: 2300 > Reynolds > 10000'.format(re))

    @property
    def nusselt_annulus(self):
        re = self.reynolds_annulus()
        if re < 2300:
            return self._get_laminal_flow_nusselt_annulus()
        elif re > 10000:
            if self.prandtl_annulus < 0.7 or self.prandtl_annulus > 160:
                raise Warning('Annulus Prandtl number is {:.4f}, which is out of range: 0.7 < Pr < 160'.format(
                    self.prandtl_annulus))
            return 0.023 * re ** 0.8 * self.prandtl_annulus ** 0.4
        else:
            raise Warning('Average Annulus fluid Reynolds number calculated as {:.2f} which is in transitional regime'
                          ' results unreliable. Acceptable range is: 2300 > Reynolds > 10000'.format(re))

    @property
    def h_annulus(self):
        hydraulic_dia_annulus = self._dia_annulus_inner_m - self._dia_pipe_outer_m
        return self.nusselt_annulus * self.k_annulus / hydraulic_dia_annulus

    @property
    def h_pipe(self):
        return self.nusselt_pipe * self.k_pipe / self._dia_pipe_inner_m

    @property
    def u_overall(self):
        # Overall heat transfer coefficent based on outside area of pipe
        return 1 / (self._dia_pipe_outer_m / (self._dia_pipe_inner_m * self.h_pipe)
                    + 1 / self.h_annulus
                    + self._dia_pipe_outer_m * np.log(self._dia_pipe_outer_m / self._dia_pipe_inner_m) / (
                                2 * self.k_wall)
                    )

    @property
    def heat_exchanger_length(self):
        return self.q_exchanged / (self.u_overall * np.pi * self._dia_pipe_outer_m * self.lmtd)

    def _get_k_fluid(self, fluid_temp_mean_kelvin, fluid_type):
        k = fluid_type['thermal_conductivity']['k_w_m_K']
        temp = fluid_type['thermal_conductivity']['temp_range_k']
        if np.max(fluid_temp_mean_kelvin) > max(temp):
            raise Warning('Temperature exceeds maximum value of {}'.format(max(temp)))
        elif np.min(fluid_temp_mean_kelvin) < min(temp):
            raise Warning('Temperature below minimum value of {}'.format(min(temp)))
        else:
            return np.interp(fluid_temp_mean_kelvin, temp, k)

    def check_pipe_temperatures_initial(self):
        if self.t_annulus_hot < self.t_annulus_cold:
            raise Warning('Annulus hot temperature must be greater than annulus cold temperature')
        elif self.t_pipe_hot < self.t_pipe_cold:
            raise Warning('Pipe hot temperature must be greater than pipe cold temperature')

    def check_pipe_temperatures_final(self):
        if self.t_annulus_hot < self.t_annulus_cold:
            raise Warning('Annulus hot temperature must be greater than annulus cold temperature')
        elif self.t_pipe_hot < self.t_pipe_cold:
            raise Warning('Pipe hot temperature must be greater than pipe cold temperature')

        if (self.t_pipe_hot > self.t_annulus_hot) != (self.t_pipe_cold > self.t_annulus_cold):
            raise Warning('Error! Heat exchanged from cold to hot! Check fluid inlet temperatures and flow rates.')

    def calculate_inlet_outlet_temperatures(self):
        if self._configuration_type == "fluid annulus hot":
            self.q_exchanged = self._mass_flow_pipe * self.cp_pipe * (self.t_pipe_hot - self.t_pipe_cold)
            self.t_annulus_hot = self.q_exchanged / (self._mass_flow_annulus * self.cp_annulus) + self.t_annulus_cold
            self.t_annulus_cold = self.t_annulus_cold
            self.t_pipe_cold = self.t_pipe_cold
            self.t_pipe_hot = self.t_pipe_hot

        elif self._configuration_type == "fluid annulus cold":
            self.q_exchanged = self._mass_flow_pipe * self.cp_pipe * (self.t_pipe_hot - self.t_pipe_cold)
            self.t_annulus_cold = self.t_annulus_hot - self.q_exchanged / (self._mass_flow_annulus * self.cp_annulus)
            self.t_annulus_hot = self.t_annulus_hot
            self.t_pipe_hot = self.t_pipe_hot
            self.t_pipe_cold = self.t_pipe_cold

        elif self._configuration_type == "fluid pipe hot":
            self.q_exchanged = self._mass_flow_annulus * self.cp_annulus * (self.t_annulus_hot - self.t_annulus_cold)
            self.t_pipe_hot = self.q_exchanged / (self._mass_flow_pipe * self.cp_pipe) + self.t_pipe_cold
            self.t_pipe_cold = self.t_pipe_cold
            self.t_annulus_hot = self.t_annulus_hot
            self.t_annulus_cold = self.t_annulus_cold

        elif self._configuration_type == "fluid pipe cold":
            self.q_exchanged = self._mass_flow_annulus * self.cp_annulus * (self.t_annulus_hot - self.t_annulus_cold)
            self.t_pipe_cold = self.t_pipe_hot - self.q_exchanged / (self._mass_flow_pipe * self.cp_pipe)

        else:
            raise Warning(
                'Unrecognised fluid inlet outlet temperature configuration, please select which temperature may vary')

    @property
    def lmtd(self):
        # log mean temperature difference
        if self._flow_configuration_type == 'counter-flow':
            if self.t_annulus_hot > self.t_pipe_hot:
                delta_t1 = self.t_annulus_hot - self.t_pipe_hot
                delta_t2 = self.t_annulus_cold - self.t_pipe_cold

            else:
                delta_t1 = self.t_pipe_hot - self.t_annulus_hot
                delta_t2 = self.t_pipe_cold - self.t_annulus_cold

        elif self._flow_configuration_type == 'parallel-flow':
            if self.t_annulus_hot > self.t_pipe_hot:
                delta_t1 = self.t_annulus_hot - self.t_pipe_cold
                delta_t2 = self.t_annulus_cold - self.t_pipe_hot
            else:
                delta_t1 = self.t_pipe_hot - self.t_annulus_cold
                delta_t2 = self.t_pipe_cold - self.t_annulus_hot
        else:
            raise Warning('Unrecognised flow configuration should be either counter-flow or parallel-flow')
        return (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)

    def reynolds_pipe(self):
        return 4 * self._mass_flow_pipe / (np.pi * self._dia_pipe_inner_m * self.visc_pipe)

    def reynolds_annulus(self):
        return 4 * self._mass_flow_annulus / (np.pi * (self._dia_annulus_inner_m + self._dia_pipe_outer_m)
                                              * self.visc_annulus)
    def _get_laminal_flow_nusselt_annulus(self):
        di_do = self._dia_pipe_outer_m/self._dia_annulus_inner_m
        if di_do < 0.05:
            raise Warning('Ratio of diameter of inner pipe to annulus is too small and not supported')
        return np.interp(di_do,
                         [0, 0.05, 0.10, 0.25, 0.50, 1.00],
                         [np.inf, 17.46, 11.56, 7.37, 5.74, 4.86])


def validate_numeric_fields(form_data, key):
    try:
        return float(form_data[key])
    except (ValueError, TypeError):
        raise Warning('Numeric input field is required for: {}'.format(key))


def main(form_data):
    numerics = ["fluid_pipe_hot_temp_c", "fluid_pipe_cold_temp_c", "fluid_pipe_flow_rate_kg_s",
                "fluid_annulus_hot_temp_c", "fluid_annulus_cold_temp_c", "fluid_annulus_flow_rate_kg_s",
                "dia_inner_annulus_mm",
                "dia_outer_pipe_mm", "dia_inner_pipe_mm", "thermal_cond_pipe_w_m_k"]

    numeric_data = {key: validate_numeric_fields(form_data, key) for key in numerics}
    heat_ex = Hx(
        annulus_fluid_data=form_data["fluid_annulus"],
        pipe_fluid_data=form_data["fluid_pipe"],
        k_wall=numeric_data["thermal_cond_pipe_w_m_k"],
        t_annulus_cold_kelvin=numeric_data["fluid_annulus_cold_temp_c"] + 273.15,
        t_annulus_hot_kelvin=numeric_data["fluid_annulus_hot_temp_c"] + 273.15,
        t_pipe_hot_kelvin=numeric_data["fluid_pipe_hot_temp_c"] + 273.15,
        t_pipe_cold_kelvin=numeric_data["fluid_pipe_cold_temp_c"] + 273.15,
        dia_annulus_inner_m=numeric_data["dia_inner_annulus_mm"] * 1e-3,
        dia_pipe_outer_m=numeric_data["dia_outer_pipe_mm"] * 1e-3,
        dia_pipe_inner_m=numeric_data["dia_inner_pipe_mm"] * 1e-3,
        mass_flow_annulus=numeric_data["fluid_annulus_flow_rate_kg_s"],
        mass_flow_pipe=numeric_data["fluid_pipe_flow_rate_kg_s"],
        temperature_configuration_type=form_data["temperature_variable"]["configuration"],
        flow_configurate_type=form_data["hx_flow_type"]["flow_type"],
    )

    return {"container_outputs": {"action": "update"},
            "main_outputs": {"action": "update"},
            "fluid_pipe_hot_temp_c_out": {"action": "update", "value": heat_ex.t_pipe_hot - 273.15},
            "fluid_pipe_cold_temp_c_out": {"action": "update", "value": heat_ex.t_pipe_cold - 273.15},
            "fluid_annulus_hot_temp_c_out": {"action": "update", "value": heat_ex.t_annulus_hot - 273.15},
            "fluid_annulus_cold_temp_c_out": {"action": "update", "value": heat_ex.t_annulus_cold - 273.15},
            "log_mean_temp": {"action": "update", "value": heat_ex.lmtd},
            "heat_exchanger_length": {"action": "update", "value": heat_ex.heat_exchanger_length},
            "mean_temp_annulus": {"action": "update", "value": heat_ex.annulus_mean_temp - 273.15},
            "mean_temp_pipe": {"action": "update", "value": heat_ex.pipe_mean_temp - 273.15},
            "reynolds_annulus": {"action": "update", "value": heat_ex.reynolds_annulus()},
            "reynolds_pipe": {"action": "update", "value": heat_ex.reynolds_pipe()},
            "prandtl_pipe": {"action": "update", "value": heat_ex.prandtl_pipe},
            "prandtl_annulus": {"action": "update", "value": heat_ex.prandtl_annulus},
            # "annulus_hydraulic_diameter_mm": {"action": "update", "value": hydraulic_dia_annulus_meters*1e3},
            "nusselt_pipe": {"action": "update", "value": heat_ex.nusselt_pipe},
            "nusselt_annulus": {"action": "update", "value": heat_ex.nusselt_annulus},
            "heat_trans_coeff_annulus": {"action": "update", "value": heat_ex.h_annulus},
            "heat_trans_coeff_pipe": {"action": "update", "value": heat_ex.h_pipe},
            "overall_heat_transfer_coeff": {"action": "update", "value": heat_ex.u_overall},
            }

