# -*- coding: utf-8 -*-
from sympl import Prognostic, get_numpy_array, combine_dimensions

c1 = None  # constant in Golaz et al. 2002 eqn 24a
c2 = None  # constant in Golaz et al. 2002 eqn 24a-b


class CHOMP(Prognostic):
    pass


def get_tendencies(
    u, v, w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal):
    tendencies = {}
    (w4, w_qn2, w_thetal2, w_qn_thetal, w_dpdz, w2_dpdz, qn_dpdz,
     thetal_dpdz) = moment_closure(
        w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal)
    turbulence_kinetic_energy = 3./2*w2

    tendencies['w2'] = -1*d_dz(w3) - 2*w2*d_dz(w) + 2*g/theta_0*w_thetav - 2/rho_0*w_dpdz


def d_dz(quantity):
    raise NotImplementedError


def moment_closure(w, w2, w3, qn, qn2, thetal, thetal2, w_qn, w_thetal, qn_thetal):
    raise NotImplementedError
