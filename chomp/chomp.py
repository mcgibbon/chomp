# -*- coding: utf-8 -*-
from sympl import (
    Prognostic, get_numpy_array, combine_dimensions, default_constants,
    restore_dimensions)
from hoc import Closure, Moment, MomentCollection
import numpy as np
from scipy.special import erf

c1 = 1.7  # constant in Golaz et al. 2002 eqn 24a
c2 = 1.04  # constant in Golaz et al. 2002 eqn 24a-b
c5 = 0.  # constant in Golaz et al. 2002 eqn 19-21
c6 = 4.85  # constant in Golaz et al. 2002 eqn 19-21
c7 = 0.8  # constant in Golaz et al. 2002 eqn 19-21
c8 = 2.73  # constant in Golaz et al. 2002 eqn 23
c11 = 0.2  # constant in Golaz et al. 2002 eqn 23
c_K = 0.548  # constant in Golaz et al. 2002 eqn. 35, as in Duynkerke and Driedonks (1987)
nu_1 = 20.  # constant in Golaz et al. 2002 eqn 24a
nu_2 = 20.  # constant in Golaz et al. 2002 eqn 24a-b
nu_6 = 30.  # constant in Golaz et al. 2002 eqn 24c


def es_from_T_Bolton(T):
    return 611.2*np.exp(17.67*(T-273.15)/(T-29.65))


def get_qs(T, p, Rd, Rv):
    epsilon_0 = Rd/Rv
    es = es_from_T_Bolton(T)
    return epsilon_0 * es/(p - (1 - epsilon_0)*es)


def get_Beta(T, Rd, Rv, Cpd, Lv):
    return Rd/Rv * (Lv/Rd) * (Lv/Cpd) / T**2


def get_Tl(T, ql, Lv, Cpd):
    # q is specific humidity
    return T - Lv/Cpd*ql

def Tl_from_thetal(thetal, p, p0, Rd, Cpd):
    return thetal*(p/p0)**Rd/Cpd

adg1 = Closure('adg1')


class HOC(Prognostic):

    closure_type = None

    def __init__(self):
        self.Cpd = default_constants[
            'heat_capacity_of_dry_air_at_constant_pressure'].to_units(
                'J kg^-1 K^-1').values.item()
        self.Rv = default_constants[
            'gas_constant_of_water_vapor'].to_units('J kg^-1 K^-1').values.item()
        self.Rd = default_constants[
            'gas_constant_of_dry_air'].to_units('J kg^-1 K^-1').values.item()
        self.Lv = default_constants[
            'latent_heat_of_vaporization_of_water'].to_units('J kg^-1')
        self.g = default_constants['gravitational_acceleration'].to_units('m s^-2')
        self.p0 = default_constants['reference_pressure'].to_units('Pa')

    def __call__(self, state):
        p = get_numpy_array(state['air_pressure'].to_units('Pa'), ['*', 'z'])
        p_interface = get_numpy_array(
            state['air_pressure_on_interface_levels'].to_units('Pa'), ['*', 'z'])
        z = get_numpy_array(
            state['height_above_reference_ellipsoid'].to_units('m'), ['*', 'z'])
        z_interface = get_numpy_array(
            state['height_above_reference_ellipsoid_on_interface_levels'
            ].to_units('m'), ['*', 'z'])
        u = get_numpy_array(state['eastward_wind'].to_units('m/s'), ['*', 'z'])
        v = get_numpy_array(state['northward_wind'].to_units('m/s'), ['*', 'z'])
        w = get_numpy_array(state['vertical_wind'].to_units('m/s'), ['*', 'z'])
        w2 = get_numpy_array(
            state['vertical_wind_variance_on_interface_levels'].to_units('m^2 s^-2'), ['*', 'z'])
        w3 = get_numpy_array(
            state['vertical_wind_skewness'].to_units('m^3 s^-3'), ['*', 'z'])
        qn = get_numpy_array(
            state['nonprecipitating_water_mixing_ratio'].to_units('kg/kg'), ['*', 'z'])
        qn2 = get_numpy_array(
            state['nonprecipitating_water_mixing_ratio_variance_on_interface_levels'
            ].to_units(''), ['*', 'z'])
        thetal = get_numpy_array(
            state['liquid_water_potential_temperature'].to_units('degK'), ['*', 'z'])
        thetal2 = get_numpy_array(
            state['liquid_water_potential_temperature_variance_on_interface_levels'
            ].to_units('degK^2'), ['*', 'z'])
        w_qn = get_numpy_array(
            state['correlation_of_vertical_wind_and_nonprecipitating_water_mixing_ratio_on_interface_levels'
                ].to_units('m/s'), ['*', 'z'])
        w_thetal = get_numpy_array(
            state['correlation_of_vertical_wind_and_liquid_water_potential_temperature_on_interface_levels'
                ].to_units('m/s'), ['*', 'z'])
        qn_thetal = get_numpy_array(
            state['correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature_on_interface_levels'
                ].to_units('degK'), ['*', 'z'])
        rho = get_numpy_array(state['air_density'].to_units('kg m^-3'), ['*', 'z'])
        tendencies, moments = get_tendencies_and_higher_order_moments(
            p, p_interface, z, z_interface, u, v, w, w2, w3, qn, qn2, thetal,
            thetal2, w_qn, w_thetal,
            qn_thetal, rho, self.g, self.p0, self.Rd, self.Rv, self.Cpd,
            self.Lv, self.closure_type)
        out_tendencies = {
            'eastward_wind': restore_dimensions(
                tendencies['u'], from_dims=['*', 'z'],
                result_like=state['eastward_wind'],
                result_attrs={'units': 'm s^-2'}),
            'northward_wind': restore_dimensions(
                tendencies['v'], from_dims=['*', 'z'],
                result_like=state['northward_wind'],
                result_attrs={'units': 'm s^-2'}),
            'nonprecipitating_water_mixing_ratio': restore_dimensions(
                tendencies['qn'], from_dims=['*', 'z'],
                result_like=state['nonprecipitating_water_mixing_ratio'],
                result_attrs={'units': 's^-1'}),
            'liquid_water_potential_temperature': restore_dimensions(
                tendencies['thetal'], from_dims=['*', 'z'],
                result_like=state['liquid_water_potential_temperature'],
                result_attrs={'units': 'degK s^-1'}),
            'vertical_wind_variance_on_interface_levels': restore_dimensions(
                tendencies['w2'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units', 'm^2 s^-3'}),
            'vertical_wind_skewness': restore_dimensions(
                tendencies['w3'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_skewness'],
                result_attrs={'units': 'm^3 s^-4'}),
            'nonprecipitating_water_mixing_ratio_variance_on_interface_levels': restore_dimensions(
                tendencies['qn2'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 's^-1'}),
            'liquid_water_potential_temperature_variance_on_interface_levels': restore_dimensions(
                tendencies['thetal2'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'K^2 s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature_on_interface_levels': restore_dimensions(
                tendencies['qn_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'K s^-1'}),
            'correlation_of_vertical_wind_and_nonprecipitating_water_mixing_ratio_on_interface_levels': restore_dimensions(
                tendencies['w_qn'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'm s^-2'}),
            'correlation_of_vertical_wind_and_liquid_water_potential_temperature_on_interface_levels': restore_dimensions(
                tendencies['w_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'm K s^-2'}),
        }
        diagnostics = {
            'vertical_wind_kurtosis_on_interface_levels': restore_dimensions(
                moments['w4'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'm^4 s^-4'}),
            'correlation_of_vertical_wind_and_nonprecipitating_water_mixing_ratio_variance': restore_dimensions(
                moments['w_qn2'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm s^-1'}),
            'correlation_of_vertical_wind_and_liquid_water_potential_temperature_variance': restore_dimensions(
                moments['w_thetal2'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm K^2 s^-1'}),
            'correlation_of_vertical_wind_and_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature': restore_dimensions(
                moments['w_qn_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm K s^-1'}),
            'correlation_of_vertical_wind_and_vertical_derivative_of_pressure_perturbation_on_interface_levels': restore_dimensions(
                moments['w_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'Pa s^-1'}),
            'correlation_of_vertical_wind_variance_and_nonprecipitating_water_mixing_ratio': restore_dimensions(
                moments['w2_qn'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm^2 s^-2'}),
            'correlation_of_vertical_wind_variance_and_liquid_water_potential_temperautre': restore_dimensions(
                moments['w2_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm^2 K s^-2'}),
            'correlation_of_vertical_wind_variance_and_vertical_derivative_of_pressure_perturbation': restore_dimensions(
                moments['w2_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm Pa s^-2'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_vertical_derivative_of_pressure_perturbation_on_interface_levels': restore_dimensions(
                moments['qn_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'Pa m^-1'}),
            'correlation_of_liquid_water_potential_temperature_and_vertical_derivative_of_pressure_perturbation_on_interface_levels': restore_dimensions(
                moments['thetal_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'K Pa m^-1'}),
            'liquid_water_mixing_ratio': restore_dimensions(
                moments['ql'], from_dims=['*', 'z'],
                result_like=state['nonprecipitating_water_mixing_ratio'],
                result_attrs={'units': 'kg/kg'}),
            'correlation_of_vertical_wind_and_liquid_water_mixing_ratio_on_interface_levels': restore_dimensions(
                moments['w_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'm s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_mixing_ratio_on_interface_levels': restore_dimensions(
                moments['qn_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'kg^2/kg^2'}),
            'correlation_of_liquid_water_potential_temperature_and_liquid_water_mixing_ratio_on_interface_levels': restore_dimensions(
                moments['thetal_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'K'}),
            'correlation_of_vertical_wind_variance_and_liquid_water_mixing_ratio': restore_dimensions(
                moments['w2_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm^2 s^-2'}),
            'virtual_potential_temperature': restore_dimensions(
                moments['thetav'], from_dims=['*', 'z'],
                result_like=state['liquid_water_potential_temperature'],
                result_attrs={'units': 'K'}),
            'correlation_of_vertical_wind_and_virtual_potential_temperature_on_interface_levels': restore_dimensions(
                moments['w_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'm K s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_virtual_potential_temperature_on_interface_levels': restore_dimensions(
                moments['qn_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'K'}),
            'correlation_of_liquid_water_potential_temperature_and_virtual_potential_temperature_on_interface_levels': restore_dimensions(
                moments['thetal_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_wind_variance'],
                result_attrs={'units': 'K^2'}),
            'correlation_of_vertical_wind_variance_and_virtual_potential_temperature': restore_dimensions(
                moments['w2_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_wind'],
                result_attrs={'units': 'm^2 K s^-2'}),
        }
        return out_tendencies, diagnostics


class CHOMP(HOC):
    closure_type = 'chomp'


class CLUBB(HOC):
    closure_type = 'adg1'


def get_tendencies_and_higher_order_moments(
    p, p_interface, z, z_interface, u, v, w, w2, w3, qn, qn2, thetal, thetal2,
    w_qn, w_thetal, qn_thetal, rho, g, p0, Rd, Rv, Cpd, Lv, closure_type='chomp'):
    epsilon_0 = Rd / Rv
    if closure_type not in ['chomp', 'adg1']:
        raise ValueError("closure_type must be one of ['chomp', 'adg1']")
    tendencies = {}
    turbulence_kinetic_energy = (3./2*w2)
    sqrt_turbulence_kinetic_energy = turbulence_kinetic_energy**0.5
    L1, L2 = get_eddy_length_scales(thetav, qn, qs, turbulence_kinetic_energy, z, g)
    tau_max = 900  # seconds, as in Golaz et al. 2002 eqn 25
    tau_1 = min(tau_max, L1/sqrt_turbulence_kinetic_energy)
    tau_2 = max(tau_max, L2/sqrt_turbulence_kinetic_energy)

    dw_dz = d_dz_mid(w)
    dqn_dz = d_dz_mid(qn)
    dthetal_dz = d_dz_mid(thetal)

    K_m = c_K*L1*sqrt_turbulence_kinetic_energy
    du_dz = d_dz_mid(u)
    dv_dz = d_dz_mid(v)
    u_w = -K_m * du_dz
    v_w = -K_m * dv_dz
    # Eddy fluxes of momentum are 0 at bottom and top of domain
    tendencies['u'] = np.zeros_like(u)
    tendencies['u'][:, 1:-1] = -d_dz_interface(u_w)
    tendencies['v'] = np.zeros_like(v)
    tendencies['v'][:, 1:-1] = -d_dz_interface(v_w)

    # unlike in CLUBB, there is no constant "a" (see Golaz et al. 2002 eqn 26)
    tau_w_w_w = tau_1
    epsilon_w_w_w = c8 / tau_w_w_w * w3

    if closure_type.lower() in ['clubb', 'adg1']:
        (tau_w_w_w, w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_qn, w2_thetal,
         w2_dpdz, qn_dpdz, thetal_dpdz, ql, w_ql, qn_ql, thetal_ql, w2_ql,
         thetav, w_thetav, qn_thetav, thetal_thetav,
         w2_thetav, C) = clubb_moment_closure(
            p, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal,
            rho, tau_1, tau_2, tau_w_w_w, dw_dz, u_w, v_w, du_dz, dv_dz, g,
            p0, Rd, Rv, Cpd, Lv, epsilon_0)
    elif closure_type.lower() is 'chomp':
        raise NotImplementedError
        # Golaz et al. 2002 eqn 33
        # we modify theta0 -> theta due to Bougealt et al. 1981b eqns 4-5
        theta = thetal - (p0/p)**(Rd/Cpd) * Lv/Cpd * ql
        thetav = theta*(1 + 0.61*qn - ql)
        constant_1 = (1 - epsilon_0)/epsilon_0*theta
        constant_2 = Lv/Cpd*(p0/p)**(Rd/Cpd) - theta/epsilon_0
        theta_interface = mid_to_interface_levels(theta, z, z_interface)
        constant_1_interface = mid_to_interface_levels(constant_1, z, z_interface)
        constant_2_interface = Lv/Cpd*(p0/p_interface)**Rd/Cpd - theta_interface/epsilon_0
        w_thetav = w_thetal + constant_1_interface*w_qn + constant_2_interface*w_ql
        qn_thetav = qn_thetal + constant_1_interface*qn2 + constant_2_interface*qn_ql
        thetal_thetav = thetal2 + constant_1_interface*qn_thetal + constant_2_interface*thetal_ql
        w2_thetav = w2_thetal + constant_1*w2_qn + constant_2*w2_ql

    # dissipation rates as defined in Golaz et al. 2002 eqn 24a-c
    epsilon_w_w = c1 / tau_1 * w2 - nu_1 * d2_dz2_interface(w2)
    epsilon_qn_qn = c2 / tau_1 * qn2 - nu_2 * d2_dz2_interface(qn2)
    epsilon_thetal_thetal = c2 / tau_1 * thetal2 - nu_2 * d2_dz2_interface(thetal2)
    epsilon_qn_thetal = c2 / tau_1 * qn_thetal - nu_2 * d2_dz2_interface(qn_thetal)
    epsilon_w_qn = -1 * nu_6 * d2_dz2_interface(w_qn)
    epsilon_w_thetal = -1 * nu_6 * d2_dz2_interface(w_thetal)


    # Golaz et al. 2002 eqns 12-18
    tendencies['w2'] = -d_dz_mid(w3) - 2*w2*dw_dz + 2*g / thetav * w_thetav - 2 / rho * w_dpdz - epsilon_w_w
    tendencies['qn2'] = -d_dz_mid(w_qn2) - 2*w_qn*dqn_dz - epsilon_qn_qn
    tendencies['thetal2'] = -d_dz_mid(w_thetal2) - 2*w_thetal*dthetal_dz - epsilon_thetal_thetal
    tendencies['qn_thetal'] = -d_dz_mid(w_qn_thetal) - w_qn*dthetal_dz - w_thetal*dqn_dz - epsilon_qn_thetal
    tendencies['w_qn'] = -d_dz_mid(w2_qn) - w2*dqn_dz - w_qn*dw_dz + g/thetav*qn_thetav - qn_dpdz/rho - epsilon_w_qn
    tendencies['w_thetal'] = -d_dz_mid(w2_thetal) - w2*dthetal_dz - w_thetal*dthetal_dz + g/thetav*thetal_thetav - thetal_dpdz/rho - epsilon_w_thetal
    tendencies['w3'] = -d_dz_interface(w4) + 3*w2*d_dz_interface(w2) - 2*w3*dw_dz + 3*g/thetav*w2_thetav - 3/rho*w2_dpdz - epsilon_w_w_w
    tendencies['qn'] = -d_dz_interface(w_qn)
    tendencies['thetal'] = -d_dz_interface(w_thetal)

    moments = {
        'w4': w4,
        'w_qn2': w_qn2,
        'w_thetal2': w_thetal2,
        'w_qn_thetal': w_qn_thetal,
        'w_dpdz': w_dpdz,
        'w2_qn': w2_qn,
        'w2_thetal': w2_thetal,
        'w2_dpdz': w2_dpdz,
        'qn_dpdz': qn_dpdz,
        'thetal_dpdz': thetal_dpdz,
        'ql': ql,
        'w_ql': w_ql,
        'qn_ql': qn_ql,
        'thetal_ql': thetal_ql,
        'w2_ql': w2_ql,
        'thetav': thetav,
        'w_thetav': w_thetav,
        'qn_thetav': qn_thetav,
        'thetal_thetav': thetal_thetav,
        'w2_thetav': w2_thetav,
    }

    return tendencies, moments


def interface_to_mid_levels(var_i, z_i, z_m):
    return (var_i[:, 1:] - var_i[:, :-1])/(z_i[:, 1:] - z_i[:, :-1]) * (z_m - z_i[:, :-1]) + var_i[:, :-1]


def mid_to_interface_levels(var_m, z_m, z_i):
    var_i = np.zeros([var_m.shape[0], var_m.shape[1]-1], dtype=var_m.dtype)
    var_i[:, 1:-1] = (var_m[:, 1:] - var_m[:, :-1])/(z_m[:, 1:] - z_m[:, :-1]) * (z_i[:, 1:-1] - z_m[:, :-1]) + var_m[:, :-1]
    var_i[:, 0] = (var_m[:, 1] - var_m[:, 0])/(z_m[:, 1] - z_m[:, 0]) * (z_i[:, 0] - z_m[:, 0]) + var_m[:, 0]
    var_i[:, -1] = (var_m[:, -1] - var_m[:, -2])/(z_m[:, -1] - z_m[:, -2]) * (z_i[:, -1] - z_m[:, -1]) + var_m[:, -1]
    return var_i


def d_dz_mid(quantity):
    """Derivative applied to quantity at mid level"""
    return_value = np.zeros(
        [quantity.shape[0], quantity.shape[1]+1], dtype=quantity.dtype)
    return_value[1:-1] = quantity[:, 1:] - quantity[:, :-1]
    # in CLUBB it is assumed quantities are linear outside of their range at
    # the top and bottom of the domain, so the derivative is constant
    return_value[0] = return_value[1]
    return_value[-1] = return_value[-2]
    return return_value


def d_dz_interface(quantity):
    return quantity[:, 1:] - quantity[:, :-1]


def d2_dz2_mid(quantity):
    return d_dz_interface(d_dz_mid(quantity))


def d2_dz2_interface(quantity):
    return d_dz_mid(d_dz_interface(quantity))


def get_eddy_length_scales(thetav, thetal, p, qn, turbulence_kinetic_energy, z,
                           g, p0, Rd, Rv, Cpd, mu=6e-4):
    """
    Calculates L1 and L2 as in Golaz et al. 2002 Section 3b. mu is fractional
    entrainment of upward plumes as defined below equation 39, in units of m^-1.
    """
    e = turbulence_kinetic_energy
    # must find heights first before lengths to implement nonlocal formulation
    z_up = np.zeros(thetav.shape)
    z_down = np.zeros(thetav.shape)
    e_remaining = np.zeros(e.shape[:1])
    done = np.zeros(e.shape[:1], dtype=np.bool)  # whether overshoot level has been found
    z_up_max = np.zeros(thetav.shape[:1])
    parcel_thetav = np.zeros(thetav.shape[:1])
    parcel_thetal = np.zeros(thetav.shape[:1])
    parcel_thetal = np.zeros(thetav.shape[:1])
    parcel_qn = np.zeros(thetav.shape[:1])
    for iz in range(thetav.shape[1]-1):
        # in CLUBB, they entrain thetal and qn for the parcel, and then calculate thetav (mixing_length.F90 line 305)
        # also they seem to use mixing ratio instead of specific humidity, should find out why
        parcel_thetav[:] = thetav[:, iz]
        parcel_thetal[:] = thetal[:, iz]
        parcel_qn[:] = qn[:, iz]
        e_remaining[:] = e[:, iz]
        done[:] = False
        for iz_plus in range(iz+1, thetav.shape[1]):
            dE_dz = -1*g*(parcel_thetav[:, iz][~done] - thetav[:, iz_plus][~done])/thetav[:, iz_plus][~done]
            dz = z[:, iz_plus][~done] - z[:, iz_plus - 1][~done]
            dE = dE_dz*dz
            assert dE > 0
            new_done = e_remaining[~done] < dE
            assert e_remaining[~done][new_done]/dE_dz[new_done] > 0
            z_up[:, iz][~done][new_done] = z[:, iz_plus-1][~done][new_done] + e_remaining[~done][new_done]/dE_dz[new_done]
            # Can only subtract energy after we're done using it for this level, and not earlier
            e_remaining[~done] -= dE[~done]
            assert np.all(e_remaining[~done] > 0)
            done[~done][new_done] = True
            mu_dz = mu * dz[~new_done]
            parcel_thetav[~done] += mu_dz * (thetav[:, iz_plus][~done] - parcel_thetav[~done])
            parcel_qn[~done] += mu_dz * (qn[:, iz_plus][~done] - parcel_qn[~done])
            # TODO: complete the latent heat effect on parcel
            Tl = Tl_from_thetal(thetal, p, p0, Rd, Cpd)
            qs = get_qs(Tl, p, Rd, Rv)  # I'd think this should be T not Tl, but Tl is what CLUBB uses (e.g. mixing_length.F90 line 294)
            supersat = parcel_qn[~done] > qs
            if np.all(done):
                break
        else:  # only triggered if for loop doesn't "break"
            z_up[:, iz][~done] = z[:, iz_plus][~done]
        new_maximum = z_up_max < z_up[:, iz]
        z_up[:, iz][~new_maximum] = z_up_max[~new_maximum]
        z_up_max[new_maximum] = z_up[:, iz][new_maximum]
    z_down_min = np.zeros(thetav.shape[:1])
    for iz in range(1, thetav.shape[1]):
        e_remaining[:] = e[:, iz]
        done[:] = False
        for iz_minus in range(iz-1, 0, -1):
            dE_dz = g*(thetav[:, iz][~done] - thetav[:, iz_plus][~done])/thetav[:, iz_plus][~done]
            dz = z[:, iz_minus][~done] - z[:, iz_minus + 1][~done]
            dE = dE_dz*dz
            assert dE > 0
            new_done = e_remaining[~done] < dE
            assert e_remaining[~done][new_done]/dE_dz[new_done] > 0
            z_down[:, iz][~done][new_done] = z[:, iz_minus+1][~done][new_done] - e_remaining[~done][new_done]/dE_dz[new_done]
            # Can only subtract energy after we're done using it for this level, and not earlier
            e_remaining[~done] -= dE[~done]
            assert np.all(e_remaining[~done] > 0)
            done[~done][new_done] = True
            if np.all(done):
                break
        else:  # only triggered if for loop doesn't "break"
            z_down[:, iz][~done] = z[:, iz_minus][~done]
        new_minimum = z_down_min > z_down[:, iz]
        z_down[~new_minimum] = z_down_min[~new_minimum]
        z_down_min[new_minimum] = z_down[new_minimum]
    L_up = z_up - z
    L_down = z - z_down
    L = np.maximum((L_up*L_down)**0.5, 20.)  # minimum value of 20m
    L1 = np.minimum(L, 400.)
    L2 = np.minimum(L, 2000.)
    return L1, L2


def adg1_moment_closure(
        p, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal,
        qn_thetal, rho, tau_1, tau_2, tau_w_w_w, dw_dz, u_w, v_w, du_dz, dv_dz, g, p0, Rd, Rv, Cpd, Lv, epsilon_0):
    moments = MomentCollection
    moments.set(Moment(w, {'w': 1}, central=False))
    moments.set(Moment(w2, {'w': 2}, central=True))
    moments.set(Moment(w3, {'w': 3}, central=True))
    moments.set(Moment(qn, {'qt': 1}, central=False))
    moments.set(Moment(qn2, {'qt': 2}, central=True))
    moments.set(Moment(thetal, {'thetal': 1}, central=False))
    moments.set(Moment(thetal2, {'thetal': 2}, central=True))
    moments.set(Moment(w_qn, {'w': 1, 'qt': 1}, central=True))
    moments.set(Moment(w_thetal, {'w': 1, 'thetal': 1}, central=True))
    moments.set(Moment(qn_thetal, {'qt': 1, 'thetal': 1}, central=True))
    return_moments = adg1.get_outputs(moments)
    w4 = return_moments.get({'w': 4}, central=True)
    w_qn2 = return_moments.get({'w': 1, 'qt': 2}, central=True)
    w_thetal2 = return_moments.get({'w': 1, 'thetal': 2}, central=True)
    w_qn_thetal = return_moments.get({'w': 1, 'qt': 1, 'thetal': 1}, central=True)
    w2_qn = return_moments.get({'w': 2, 'qt': 1}, central=True)
    w2_thetal = return_moments.get({'w': 2, 'thetal': 1}, central=True)
    w_ql = return_moments.get({'w': 1, 'ql': 1}, central=True)
    qn_ql = return_moments.get({'qt': 1, 'ql': 1}, central=True)
    thetal_ql = return_moments.get({'thetal': 1, 'ql': 1}, central=True)
    w2_ql = return_moments.get({'w': 2, 'ql': 1}, central=True)
    ql = return_moments.get({'ql': 1}, central=False)
    C = return_moments.get({'C': 1}, central=False)

    thetav, w_thetav, qn_thetav, thetal_thetav, w2_thetav = get_thetav_moments(
        p, thetal, qn, ql, w_thetal, w_qn, w_ql, qn_thetal, qn2, qn_ql, thetal2,
    thetal_ql, w2_thetal, w2_qn, w2_ql, p0, Lv, Rd, Cpd, epsilon_0)

    w_dpdz = -rho/2. * (
        - c5 * (-2*w2*dw_dz + 2*g/thetav*w_thetav) +
        2./3*c5*(g/thetav*w_thetav - u_w))
    qn_dpdz = -rho * (
        -c6/tau_2*w_qn - c7*(-w_qn*dw_dz + g/thetav*qn_thetav)
    )
    thetal_dpdz = -rho * (
        -c6/tau_2*w_thetal - c7 * (-w_thetal*dw_dz + g/thetav*thetal_thetav)
    )
    # yeah, this isn't a real moment, but this was the easiest way to get the
    # value of a which we need for tau_w_w_w without re-calculating it
    a = return_moments.get({'a': 1}, central=False)
    tau_w_w_w[a < 0.05] = tau_w_w_w[a < 0.05]/(1 + 3*(1 - (a[a < 0.05]-0.01)/0.04))
    tau_w_w_w[a > 0.05] = tau_w_w_w[a > 0.05]/(1 + 3*(1 - (0.99 - a[a > 0.05])/0.04))
    w2_dpdz = -rho/3. * (
        -c8/tau_w_w_w*w3 - c11*(-2*w3*dw_dz + 3*g/thetav*w2_thetav))
    return (
        tau_w_w_w, w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_qn, w2_thetal,
        w2_dpdz, qn_dpdz, thetal_dpdz, ql, w_ql, qn_ql, thetal_ql, w2_ql, thetav,
        w_thetav, qn_thetav, thetal_thetav, w2_thetav, C)


def clubb_moment_closure(
        p, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal,
        qn_thetal, rho, tau_1, tau_2, tau_w_w_w, dw_dz, u_w, v_w, du_dz, dv_dz, g, p0, Rd, Rv, Cpd, Lv):
    """
    Closure is taken from Larson and Golaz (2005). Terms are as they appear in
    that paper. Except thetal3 and qn3 are closed as in Golaz et al. 2002.
    """
    Skw = w3/(w2**(3./2))

    thetal3 = np.zeros_like(thetal2)
    qn3 = 1.2*Skw*qn2**(3./2)

    sigma_w_tilde = 0.4**0.5
    sigma_w = sigma_w_tilde*w2**0.5
    Sk_thetal = thetal3/(thetal2**(3./2))
    Sk_qn = qn3/(qn2**(3./2))
    hat_factor = 1./(1. - sigma_w_tilde**2)**0.5
    Skw_hat = Skw * hat_factor
    a = 0.5*(1 - Skw*(4*(1-sigma_w_tilde**2)**3 + Skw**2)**(-0.5))
    a[a < 0.01] = 0.01
    a[a > 0.99] = 0.99

    w_1_hat = ((1-a)/a)**0.5
    w_2_hat = -1*(a/(1-a))**0.5

    c_w_thetal_hat = hat_factor*w_thetal/(w2*thetal2)**0.5
    c_w_qn_hat = hat_factor*w_qn/(w2*qn2)**0.5

    # eqn 24-25
    thetal_1_tilde = -1*c_w_thetal_hat/w_2_hat
    thetal_2_tilde = -1*c_w_thetal_hat/w_1_hat
    qn_1_tilde = -1*c_w_qn_hat/w_2_hat
    qn_2_tilde = -1*c_w_qn_hat/w_1_hat

    # eqn 26-27
    sigma_thetal_1_tilde = (
        (1 - c_w_thetal_hat**2) +
        ((1-a)/a)**0.5 * (Sk_thetal - c_w_thetal_hat**3 * Skw_hat)/(3*c_w_thetal_hat))**0.5
    sigma_thetal_2_tilde = (
        (1 - c_w_thetal_hat**2) +
        -1*(a/(1-a))**0.5 * (Sk_thetal - c_w_thetal_hat**3 * Skw_hat)/(3*c_w_thetal_hat))**0.5
    sigma_qn_1_tilde = (
        (1 - c_w_qn_hat**2) +
        ((1-a)/a)**0.5 * (Sk_qn - c_w_qn_hat**3 * Skw_hat)/(3*c_w_qn_hat))**0.5
    sigma_qn_2_tilde = (
        (1 - c_w_qn_hat**2) +
        -1*(a/(1-a))**0.5 * (Sk_qn - c_w_qn_hat**3 * Skw_hat)/(3*c_w_qn_hat))**0.5

    c_qn_thetal = qn_thetal/(qn2*thetal2)**0.5  # eqn 18

    # eqn 28
    r_qn_thetal = (c_qn_thetal - c_w_qn_hat*c_w_thetal_hat)/(a*sigma_qn_1_tilde*sigma_thetal_1_tilde + (1-a)*sigma_qn_2_tilde*sigma_thetal_2_tilde)

    w4 = w2**2/hat_factor**4 * (
        a * (w_1_hat**4 + 6*w_1_hat**2*sigma_w_tilde**2*hat_factor**2 + 3*sigma_w_tilde**4*hat_factor**4) +
        (1-a) * (w_2_hat**4 + 6*w_2_hat**2*sigma_w_tilde**2*hat_factor**2 + 3*sigma_w_tilde**4*hat_factor**4)
    )  # eqn 29

    w2_thetal = hat_factor**2/(w2*thetal2**0.5) * (
        a * (w_1_hat**2 + sigma_w_tilde**2*hat_factor**2)*thetal_1_tilde +
        (1-a) * (w_2_hat**2 + sigma_w_tilde**2*hat_factor**2)*thetal_2_tilde
    )  # eqn 30

    w_thetal2 = hat_factor/(w2**0.5*thetal2) * (
        a*w_1_hat*(thetal_1_tilde**2 + sigma_thetal_1_tilde**2) +
        (1-a)*w_2_hat*(thetal_2_tilde**2 + sigma_thetal_2_tilde**2)
    )  # eqn 31

    w2_qn = hat_factor ** 2 / (w2 * qn2 ** 0.5) * (
        a * (w_1_hat ** 2 + sigma_w_tilde ** 2 * hat_factor ** 2) * qn_1_tilde +
        (1 - a) * (w_2_hat ** 2 + sigma_w_tilde ** 2 * hat_factor ** 2) * qn_2_tilde
    )  # eqn 30

    w_qn2 = hat_factor / (w2 ** 0.5 * qn2) * (
        a * w_1_hat * (qn_1_tilde ** 2 + sigma_qn_1_tilde ** 2) +
        (1 - a) * w_2_hat * (qn_2_tilde ** 2 + sigma_qn_2_tilde ** 2)
    )  # eqn 31

    w_qn_thetal = hat_factor/(w2**0.5*qn2**0.5*thetal2**0.5) * (
        a*w_1_hat*(qn_1_tilde*thetal_1_tilde + r_qn_thetal*sigma_qn_1_tilde*sigma_thetal_1_tilde) +
        (1-a)*w_2_hat*(qn_2_tilde*thetal_2_tilde + r_qn_thetal*sigma_qn_2_tilde*sigma_thetal_2_tilde)
    )

    sigma_thetal_1 = sigma_thetal_1_tilde*thetal2**0.5
    sigma_thetal_2 = sigma_thetal_2_tilde*thetal2**0.5
    sigma_qn_1 = sigma_qn_1_tilde*qn2**0.5
    sigma_qn_2 = sigma_qn_2_tilde*qn2**0.5
    thetal_1 = thetal_1_tilde*thetal2**0.5 + thetal
    thetal_2 = thetal_2_tilde*thetal2**0.5 + thetal
    qn_1 = qn_1_tilde*qn2**0.5 + qn
    qn_2 = qn_2_tilde*qn2**0.5 + qn
    w_1 = w_1_hat/hat_factor*w2**0.5 + w
    w_2 = w_2_hat/hat_factor*w2**0.5 + w

    C_1, ql_1, thetal_ql_1, qn_ql_1 = get_values_within_gaussian(
        thetal_1, qn_1, sigma_thetal_1, sigma_qn_1, r_qn_thetal, p, p0, Rd, Rv,
        Cpd, Lv)
    C_2, ql_2, thetal_ql_2, qn_ql_2 = get_values_within_gaussian(
        thetal_2, qn_2, sigma_thetal_2, sigma_qn_2, r_qn_thetal, p, p0, Rd, Rv,
        Cpd, Lv)

    C = a*C_1 + (1-a)*C_2
    ql = a*ql_1 + (1-a)*ql_2
    w_ql = a*(w_1 - w)*ql_1 + (1-a)*(w_2 - w)*ql_2
    w2_ql = (a*((w_1 - w)**2 + sigma_w**2)*(ql_1 - ql) +
             (1-a)*((w_2 - w)**2 + sigma_w**2)*(ql_2 - ql))
    thetal_ql = (a*((thetal_1 - thetal)*ql_1 + thetal_ql_1) +
                 (1-a)*((thetal_2 - thetal)*ql_2 + thetal_ql_2))
    qn_ql = a*((qn_1 - qn)*ql_1 + qn_ql_1) + (1-a)*((qn_2 - qn)*ql_2 + qn_ql_2)

    thetav, w_thetav, qn_thetav, thetal_thetav, w2_thetav = get_thetav_moments(
        p, thetal, qn, ql, w_thetal, w_qn, w_ql, qn_thetal, qn2, qn_ql, thetal2,
    thetal_ql, w2_thetal, w2_qn, w2_ql, p0, Lv, Rd, Cpd)

    w_dpdz = -rho/2. * (
        - c5 * (-2*w2*dw_dz + 2*g/thetav*w_thetav) +
        2./3*c5*(g/thetav*w_thetav - u_w))
    qn_dpdz = -rho * (
        -c6/tau_2*w_qn - c7*(-w_qn*dw_dz + g/thetav*qn_thetav)
    )
    thetal_dpdz = -rho * (
        -c6/tau_2*w_thetal - c7 * (-w_thetal*dw_dz + g/thetav*thetal_thetav)
    )
    tau_w_w_w[a < 0.05] = tau_w_w_w[a < 0.05]/(1 + 3*(1 - (a[a < 0.05]-0.01)/0.04))
    tau_w_w_w[a > 0.05] = tau_w_w_w[a > 0.05]/(1 + 3*(1 - (0.99 - a[a > 0.05])/0.04))
    w2_dpdz = -rho/3. * (
        -c8/tau_w_w_w*w3 - c11*(-2*w3*dw_dz + 3*g/thetav*w2_thetav))
    return (
        tau_w_w_w, w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_qn, w2_thetal,
        w2_dpdz, qn_dpdz, thetal_dpdz, ql, w_ql, qn_ql, thetal_ql, w2_ql, thetav,
        w_thetav, qn_thetav, thetal_thetav, w2_thetav, C)


def get_values_within_gaussian(
        thetal_bar, qn_bar, sigma_thetal, sigma_qn, r_qn_thetal,
        p, p0, Rd, Rv, Cpd, Lv):
    Tl = Tl_from_thetal(thetal_bar, p, p0, Rd, Cpd)
    Beta = get_Beta(Tl, Rd, Rv, Cpd, Lv)
    qs = get_qs(Tl, p, Rd, Rv)
    c_qn = 1./(1 + Beta*qs)
    c_thetal = (1 + Beta*qn_bar)/(1 + Beta*qs)**2*Cpd/Lv*Beta*qs*(p/p0)**(Rd/Cpd)
    sigma_s = (
        c_thetal**2*sigma_thetal**2 +
        c_qn**2*sigma_qn**2 -
        2*c_thetal*sigma_thetal*c_qn*sigma_qn*r_qn_thetal)**0.5
    s = qn_bar - qs*(1 + Beta*qn_bar)/(1 + Beta*qs)
    C = 0.5*(1 + erf(s/(2**0.5 * sigma_s)))
    ql = s*C + sigma_s/(2*np.pi)**0.5*np.exp(-0.5*(s/sigma_s)**2)
    thetal_ql = sigma_thetal*(c_qn*sigma_qn*r_qn_thetal - c_thetal*sigma_thetal)*C
    qn_ql = sigma_qn*(c_qn*sigma_qn - c_thetal*sigma_thetal*r_qn_thetal)*C
    return C, ql, thetal_ql, qn_ql

def get_thetav_moments(
        p, thetal, qn, ql, w_thetal, w_qn, w_ql, qn_thetal, qn2, qn_ql, thetal2,
    thetal_ql, w2_thetal, w2_qn, w2_ql, p0, Lv, Rd, Cpd, epsilon_0):
    rl = ql / (1 - ql)
    rn = qn / (1 - qn)
    # Golaz et al. 2002 eqn 33
    # we modify theta0 -> theta due to Bougealt et al. 1981b eqns 4-5
    theta = thetal - Lv/Cpd * rl
    thetav = theta*(1 + 0.61*rn - rl)
    constant_1 = (1 - epsilon_0)/epsilon_0*theta
    constant_2 = Lv/Cpd*(p0/p)**(Rd/Cpd) - theta/epsilon_0
    w_thetav = w_thetal + constant_1*w_qn + constant_2*w_ql
    qn_thetav = qn_thetal + constant_1*qn2 + constant_2*qn_ql
    thetal_thetav = thetal2 + constant_1*qn_thetal + constant_2*thetal_ql
    w2_thetav = w2_thetal + constant_1*w2_qn + constant_2*w2_ql
    return thetav, w_thetav, qn_thetav, thetal_thetav, w2_thetav
