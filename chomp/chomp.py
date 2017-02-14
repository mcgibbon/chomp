# -*- coding: utf-8 -*-
from sympl import Prognostic, get_numpy_array, combine_dimensions

c1 = None  # constant in Golaz et al. 2002 eqn 24a
c2 = None  # constant in Golaz et al. 2002 eqn 24a-b
c8 = None  # constant in Golaz et al. 2002 eqn 23
nu_6 = None  # constant in Golaz et al. 2002 eqn 24c


class CHOMP(Prognostic):
    pass


def get_tendencies(
    u, v, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal):
    tendencies = {}
    (w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_dpdz, qn_dpdz,
     thetal_dpdz) = moment_closure(
        w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal)
    sqrt_turbulence_kinetic_energy = (3./2*w2)**0.5
    L1, L2 = get_eddy_length_scales()
    tau_max = 900  # seconds, as in Golaz et al. 2002 eqn 25
    tau_1 = min(tau_max, L1/sqrt_turbulence_kinetic_energy)
    tau_2 = max(tau_max, L2/sqrt_turbulence_kinetic_energy)

    # dissipation rates as defined in Golaz et al. 2002 eqn 24a-c
    epsilon_w_w = c1 / tau_1 * w2 - nu_1 * d2_dz2(w2)
    epsilon_qn_qn = c2 / tau_1 * qn2 - nu_2 * d2_dz2(qn2)
    epsilon_thetal_thetal = c2 / tau_1 * thetal2 - nu_2 * d2_dz2(thetal2)
    epsilon_qn_thetal = c2 / tau_1 * qn_thetal - nu_2 * d2_dz2(qn_thetal)
    epsilon_w_qn = -1 * nu_6 * d2_dz2(w_qn)
    epsilon_w_thetal = -1 * nu_6 * d2_dz2(w_thetal)

    # unlike in CLUBB, there is no constant "a" (see Golaz et al. 2002 eqn 26)
    tau_w_w_w = tau_1
    epsilon_w_w_w = c8 / tau_w_w_w * w3

    dw_dz = d_dz(w)
    dqn_dz = d_dz(qn)
    dthetal_dz = d_dz(thetal)

    # Golaz et al. 2002 eqns 12-18
    tendencies['w2'] = - d_dz(w3) - 2*w2*dw_dz + 2*g/theta_0*w_thetav - 2/rho_0*w_dpdz - epsilon_w_w
    tendencies['qn2'] = - d_dz(w_qn2) - 2*w_qn*dqn_dz - epsilon_qn_qn
    tendencies['thetal2'] = - d_dz(w_thetal2) - 2*w_thetal*dthetal_dz - epsilon_thetal_thetal
    tendencies['qn_thetal'] = - d_dz(w_qn_thetal) - w_qn*dthetal_dz - w_thetal*dqn_dz - epsilon_qn_thetal
    tendencies['w_qn'] = - d_dz(w2_qn) - w2*dqn_dz - w_qn*dw_dz + g/theta_0*qn_thetav - qn_dpdz/rho_0 - epsilon_w_qn
    tendencies['w_thetal'] = - d_dz(w2_thetal) - w2_dthetal_dz - w_thetal*dthetal_dz + g/theta_0*thetal_thetav - thetal_dpdz/rho_0 - epsilon_w_thetal
    tendencies['w3'] = - d_dz(w4) + e*w2*d_dz(w2) - 2*w3*dw_dz + 3*g/theta_0*w2_thetav - 3/rho_0*w2_dpdz - epsilon_w_w_w


def d_dz(quantity):
    return quantity[:, 1:] - quantity[:, :-1]


def d2_dz2(quantity):
    raise NotImplementedError


def get_eddy_length_scales():
    # Golaz et al. 2002 Section 3b
    raise NotImplementedError


def moment_closure(w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal):
    raise NotImplementedError
