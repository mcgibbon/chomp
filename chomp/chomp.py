# -*- coding: utf-8 -*-
from sympl import (
    Prognostic, get_numpy_array, combine_dimensions, default_constants,
    restore_dimensions)
from hoc import Closure, Moment, MomentCollection
import numpy as np

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
        u = get_numpy_array(state['eastward_wind'].to_units('m/s'), ['*', 'z'])
        v = get_numpy_array(state['northward_wind'].to_units('m/s'), ['*', 'z'])
        w = get_numpy_array(state['vertical_wind'].to_units('m/s'), ['*', 'z'])
        w2 = get_numpy_array(
            state['vertical_wind_variance'].to_units('m^2 s^-2'), ['*', 'z'])
        w3 = get_numpy_array(
            state['vertical_wind_skewness'].to_units('m^3 s^-3'), ['*', 'z'])
        qn = get_numpy_array(
            state['nonprecipitating_water_mixing_ratio'].to_units('kg/kg'), ['*', 'z'])
        qn2 = get_numpy_array(
            state['nonprecipitating_water_mixing_ratio_variance'].to_units(''), ['*', 'z'])
        thetal = get_numpy_array(
            state['liquid_water_potential_temperature'].to_units('degK'), ['*', 'z'])
        thetal2 = get_numpy_array(
            state['liquid_water_potential_temperature_variance'].to_units('degK^2'), ['*', 'z'])
        w_qn = get_numpy_array(
            state['correlation_of_vertical_wind_and_nonprecipitating_water_mixing_ratio'
                ].to_units('m/s'), ['*', 'z'])
        w_thetal = get_numpy_array(
            state['correlation_of_vertical_wind_and_liquid_water_potential_temperature'
                ].to_units('m/s'), ['*', 'z'])
        qn_thetal = get_numpy_array(
            state['correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature'
                ].to_units('degK'), ['*', 'z'])
        rho = get_numpy_array(state['air_density'].to_units('kg m^-3'), ['*', 'z'])
        tendencies, moments = get_tendencies_and_higher_order_moments(
            p, u, v, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal,
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
            'vertical_velocity_variance': restore_dimensions(
                tendencies['w2'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units', 'm^2 s^-3'}),
            'vertical_velocity_skewness': restore_dimensions(
                tendencies['w3'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_skewness'],
                result_attrs={'units': 'm^3 s^-4'}),
            'nonprecipitating_water_mixing_ratio_variance': restore_dimensions(
                tendencies['qn2'], from_dims=['*', 'z'],
                result_like=state['nonprecipitating_water_mixing_ratio_variance'],
                result_attrs={'units': 's^-1'}),
            'liquid_water_potential_temperature_variance': restore_dimensions(
                tendencies['thetal2'], from_dims=['*', 'z'],
                result_like=state['liquid_water_potential_temperature_variance'],
                result_attrs={'units': 'K^2 s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature': restore_dimensions(
                tendencies['qn_thetal'], from_dims=['*', 'z'],
                result_like=state['correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature'],
                result_attrs={'units': 'K s^-1'}),
            'correlation_of_vertical_velocity_and_nonprecipitating_water_mixing_ratio': restore_dimensions(
                tendencies['w_qn'], from_dims=['*', 'z'],
                result_like=state['correlation_of_vertical_velocity_and_nonprecipitating_water_mixing_ratio'],
                result_attrs={'units': 'm s^-2'}),
            'correlation_of_vertical_velocity_and_liquid_water_potential_temperature': restore_dimensions(
                tendencies['w_thetal'], from_dims=['*', 'z'],
                result_like=state['correlation_of_vertical_velocity_and_liquid_water_potential_temperature'],
                result_attrs={'units': 'm K s^-2'}),
        }
        diagnostics = {
            'vertical_velocity_kurtosis': restore_dimensions(
                moments['w4'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'm^4 s^-4'}),
            'correlation_of_vertical_velocity_and_nonprecipitating_water_mixing_ratio_variance': restore_dimensions(
                moments['w_qn2'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm s^-1'}),
            'correlation_of_vertical_velocity_and_liquid_water_potential_temperature_variance': restore_dimensions(
                moments['w_thetal2'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm K^2 s^-1'}),
            'correlation_of_vertical_velocity_and_nonprecipitating_water_mixing_ratio_and_liquid_water_potential_temperature': restore_dimensions(
                moments['w_qn_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm K s^-1'}),
            'correlation_of_vertical_velocity_and_vertical_derivative_of_pressure_perturbation': restore_dimensions(
                moments['w_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'Pa s^-1'}),
            'correlation_of_vertical_velocity_variance_and_nonprecipitating_water_mixing_ratio': restore_dimensions(
                moments['w2_qn'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm^2 s^-2'}),
            'correlation_of_vertical_velocity_variance_and_liquid_water_potential_temperautre': restore_dimensions(
                moments['w2_thetal'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm^2 K s^-2'}),
            'correlation_of_vertical_velocity_variance_and_vertical_derivative_of_pressure_perturbation': restore_dimensions(
                moments['w2_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm Pa s^-2'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_vertical_derivative_of_pressure_perturbation': restore_dimensions(
                moments['qn_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'Pa m^-1'}),
            'correlation_of_liquid_water_potential_temperature_and_vertical_derivative_of_pressure_perturbation': restore_dimensions(
                moments['thetal_dpdz'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'K Pa m^-1'}),
            'liquid_water_mixing_ratio': restore_dimensions(
                moments['ql'], from_dims=['*', 'z'],
                result_like=state['nonprecipitating_water_mixing_ratio'],
                result_attrs={'units': 'kg/kg'}),
            'correlation_of_vertical_velocity_and_liquid_water_mixing_ratio': restore_dimensions(
                moments['w_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'm s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_liquid_water_mixing_ratio': restore_dimensions(
                moments['qn_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'kg^2/kg^2'}),
            'correlation_of_liquid_water_potential_temperature_and_liquid_water_mixing_ratio': restore_dimensions(
                moments['thetal_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'K'}),
            'correlation_of_vertical_velocity_variance_and_liquid_water_mixing_ratio': restore_dimensions(
                moments['w2_ql'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity'],
                result_attrs={'units': 'm^2 s^-2'}),
            'virtual_potential_temperature': restore_dimensions(
                moments['thetav'], from_dims=['*', 'z'],
                result_like=state['liquid_water_potential_temperature'],
                result_attrs={'units': 'K'}),
            'correlation_of_vertical_velocity_and_virtual_potential_temperature': restore_dimensions(
                moments['w_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                restult_attrs={'units': 'm K s^-1'}),
            'correlation_of_nonprecipitating_water_mixing_ratio_and_virtual_potential_temperature': restore_dimensions(
                moments['qn_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'K'}),
            'correlation_of_liquid_water_potential_temperature_and_virtual_potential_temperature': restore_dimensions(
                moments['thetal_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_variance'],
                result_attrs={'units': 'K^2'}),
            'correlation_of_vertical_velocity_variance_and_virtual_potential_temperature': restore_dimensions(
                moments['w2_thetav'], from_dims=['*', 'z'],
                result_like=state['vertical_velocity_skewness'],
                result_attrs={'units': 'm^2 K s^-2'}),
        }
        return out_tendencies, diagnostics


class CHOMP(HOC):
    closure_type = 'chomp'


class CLUBB(HOC):
    closure_type = 'adg1'


def get_tendencies_and_higher_order_moments(
    p, u, v, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal, rho,
    g, p0, Rd, Rv, Cpd, Lv, closure_type='chomp'):
    epsilon_0 = Rd / Rv
    if closure_type not in ['chomp', 'adg1']:
        raise ValueError("closure_type must be one of ['chomp', 'adg1']")
    tendencies = {}
    sqrt_turbulence_kinetic_energy = (3./2*w2)**0.5
    L1, L2 = get_eddy_length_scales()
    tau_max = 900  # seconds, as in Golaz et al. 2002 eqn 25
    tau_1 = min(tau_max, L1/sqrt_turbulence_kinetic_energy)
    tau_2 = max(tau_max, L2/sqrt_turbulence_kinetic_energy)

    dw_dz = d_dz(w)
    dqn_dz = d_dz(qn)
    dthetal_dz = d_dz(thetal)

    K_m = c_K*L1*sqrt_turbulence_kinetic_energy
    u_w = -K_m * d_dz(u)
    v_w = -K_m * d_dz(v)
    # Eddy fluxes of momentum are 0 at bottom and top of domain
    tendencies['u'] = np.zeros_like(u)
    tendencies['u'][:, 1:-1] = -d_dz(u_w)
    tendencies['v'] = np.zeros_like(v)
    tendencies['v'][:, 1:-1] = -d_dz(v_w)

    # unlike in CLUBB, there is no constant "a" (see Golaz et al. 2002 eqn 26)
    tau_w_w_w = tau_1
    epsilon_w_w_w = c8 / tau_w_w_w * w3

    if closure_type is 'adg1':
        (tau_w_w_w, w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_qn, w2_thetal,
         w2_dpdz, qn_dpdz, thetal_dpdz, ql, w_ql, qn_ql, thetal_ql, w2_ql,
         thetav, w_thetav, qn_thetav, thetal_thetav, w2_thetav) = adg1_moment_closure(
            p, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal,
            rho, tau_1, tau_2, tau_w_w_w, dw_dz, u_w, v_w, d_dz(u), d_dz(v), g,
            p0, Rd, Cpd, Lv, epsilon_0)
    elif closure_type is 'chomp':
        raise NotImplementedError

    # dissipation rates as defined in Golaz et al. 2002 eqn 24a-c
    epsilon_w_w = c1 / tau_1 * w2 - nu_1 * d2_dz2(w2)
    epsilon_qn_qn = c2 / tau_1 * qn2 - nu_2 * d2_dz2(qn2)
    epsilon_thetal_thetal = c2 / tau_1 * thetal2 - nu_2 * d2_dz2(thetal2)
    epsilon_qn_thetal = c2 / tau_1 * qn_thetal - nu_2 * d2_dz2(qn_thetal)
    epsilon_w_qn = -1 * nu_6 * d2_dz2(w_qn)
    epsilon_w_thetal = -1 * nu_6 * d2_dz2(w_thetal)

    # Golaz et al. 2002 eqn 33
    # we modify theta0 -> theta due to Bougealt et al. 1981b eqns 4-5
    theta = thetal - (p0/p)**(Rd/Cpd) * Lv/Cpd * ql
    thetav = theta*(1 + 0.61*qn - ql)
    constant_1 = (1 - epsilon_0)/epsilon_0*theta
    constant_2 = Lv/Cpd*(p0/p)**(Rd/Cpd) - theta/epsilon_0
    w_thetav = w_thetal + constant_1*w_qn + constant_2*w_ql
    qn_thetav = qn_thetal + constant_1*qn2 + constant_2*qn_ql
    thetal_thetav = thetal2 + constant_1*qn_thetal + constant_2*thetal_ql
    w2_thetav = w2_thetal + constant_1*w2_qn + constant_2*w2_ql

    # Golaz et al. 2002 eqns 12-18
    tendencies['w2'] = -d_dz(w3) - 2*w2*dw_dz + 2*g / thetav * w_thetav - 2 / rho * w_dpdz - epsilon_w_w
    tendencies['qn2'] = -d_dz(w_qn2) - 2*w_qn*dqn_dz - epsilon_qn_qn
    tendencies['thetal2'] = -d_dz(w_thetal2) - 2*w_thetal*dthetal_dz - epsilon_thetal_thetal
    tendencies['qn_thetal'] = -d_dz(w_qn_thetal) - w_qn*dthetal_dz - w_thetal*dqn_dz - epsilon_qn_thetal
    tendencies['w_qn'] = -d_dz(w2_qn) - w2*dqn_dz - w_qn*dw_dz + g/thetav*qn_thetav - qn_dpdz/rho - epsilon_w_qn
    tendencies['w_thetal'] = -d_dz(w2_thetal) - w2*dthetal_dz - w_thetal*dthetal_dz + g/thetav*thetal_thetav - thetal_dpdz/rho - epsilon_w_thetal
    tendencies['w3'] = -d_dz(w4) + 3*w2*d_dz(w2) - 2*w3*dw_dz + 3*g/thetav*w2_thetav - 3/rho*w2_dpdz - epsilon_w_w_w
    tendencies['qn'] = -d_dz(w_qn)
    tendencies['thetal'] = -d_dz(w_thetal)

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




def d_dz(quantity):
    return quantity[:, 1:] - quantity[:, :-1]


def d2_dz2(quantity):
    return d_dz(d_dz(quantity))


def get_eddy_length_scales():
    # Golaz et al. 2002 Section 3b
    raise NotImplementedError


def adg1_moment_closure(
        p, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal,
        qn_thetal, rho, tau_1, tau_2, tau_w_w_w, dw_dz, u_w, v_w, du_dz, dv_dz, g, p0, Rd, Cpd, Lv, epsilon_0):
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
        w_thetav, qn_thetav, thetal_thetav, w2_thetav)


def get_thetav_moments(
        p, thetal, qn, ql, w_thetal, w_qn, w_ql, qn_thetal, qn2, qn_ql, thetal2,
    thetal_ql, w2_thetal, w2_qn, w2_ql, p0, Lv, Rd, Cpd, epsilon_0):
    # Golaz et al. 2002 eqn 33
    # we modify theta0 -> theta due to Bougealt et al. 1981b eqns 4-5
    theta = thetal - (p0/p)**(Rd/Cpd) * Lv/Cpd * ql
    thetav = theta*(1 + 0.61*qn - ql)
    constant_1 = (1 - epsilon_0)/epsilon_0*theta
    constant_2 = Lv/Cpd*(p0/p)**(Rd/Cpd) - theta/epsilon_0
    w_thetav = w_thetal + constant_1*w_qn + constant_2*w_ql
    qn_thetav = qn_thetal + constant_1*qn2 + constant_2*qn_ql
    thetal_thetav = thetal2 + constant_1*qn_thetal + constant_2*thetal_ql
    w2_thetav = w2_thetal + constant_1*w2_qn + constant_2*w2_ql
    return thetav, w_thetav, qn_thetav, thetal_thetav, w2_thetav
