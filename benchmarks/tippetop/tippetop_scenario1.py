
# This file implements the tippetop example, 
# see Section 10.6 of "A nonsmooth generalized-alpha method for 
# mechanical systems with frictional contact", Giuseppe Capobianco, 
# Jonas Harsch, Simon R. Eugster, Remco I. Leine
# Int J Numer Methods Eng. 2021; 1– 30. https://doi.org/10.1002/nme.6801
#
#
# Stuttgart, November 2021                                  G.Capobianco

# ---- we need a better solution to this
import sys
from os import path
here = path.abspath(path.dirname(__file__))
root = path.abspath(path.dirname(path.dirname(here)))
sys.path.append(here)
sys.path.append(root)
# ==== we need a better solution to this

import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

from tippetop_system import Tippetop_Leine2003, Tippetop_quaternion

from solvers.moreau import Moreau
from solvers.generalized_alpha import Generalized_alpha

if __name__ == "__main__":
  
    # show simulation results
    show_plots = True

    # Units: kg, m, s
    # Dynamics:
    m = 6e-3 # kg
    I1 = 8e-7 # kg m2 # = I_2 # Leine2013
    I3 = 7e-7 # kg m2
    g = 9.81 # kg m / s2
    # Geometry:
    a1 = 3e-3 # m
    a2 = 1.6e-2 # m
    r1 = 1.5e-2 # m
    r2 = 5e-3 # m

    mu = 0.3   # = mu1 = mu2
    eN = 0     # = eN1 = eN2
    eF = 0
    R = 5e-4   # m # = R1 = R2
    prox_r = 0.001

    # Initial conditions
    # all zeros exept:
    # Leine2003
    z0 = 1.2015e-2 # m
    theta0 = 0.1 # rad
    psi_dot0 = 180 # rad / s

    # top_g = Tippetop_Leine2003( m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0)
    top_g = Tippetop_quaternion(m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0)
    top_m = Tippetop_Leine2003( m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0)
    # top_m = Tippetop_quaternion( m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0)

    t0 = 0
    t1 = 8
    
    dt = 1e-3
    gen_al = Generalized_alpha(top_g, t0, t1, dt, rho_inf=0.5, newton_tol=1.0e-6)
    sol_g = gen_al.solve()

    t_g = sol_g[0]
    q_g = sol_g[1]
    u_g = sol_g[2]
    # a_g = sol_g[3]
    # La_N_g = sol_g[10]
    # la_N_g = sol_g[11]
    # La_F_g = sol_g[12]
    # la_F_g = sol_g[13]
    # P_N_g = sol_g[14]
    # P_F_g = sol_g[15]

    dt = 1e-4
    moreau = Moreau(top_m, t0, t1, dt, fix_point_tol=1e-6)
    sol_m = moreau.solve()

    t_m = sol_m[0]
    q_m = sol_m[1]
    u_m = sol_m[2]
    P_N_m = sol_m[5]
    P_F_m = sol_m[6]

    nt_g = len(t_g)

    # gaps and theta
    nt_m = len(t_m)
    g_N_g = np.zeros((nt_g, top_g.nla_N))
    theta_g = np.zeros(nt_g)
    g_N_m = np.zeros((nt_m, top_m.nla_N))
    theta_m = np.zeros(nt_m)

    for i, ti in enumerate(t_g):
        theta_g[i] = top_g.theta(ti, q_g[i])
        g_N_g[i] = top_g.g_N(ti, q_g[i])

    for i, ti in enumerate(t_m):
        theta_m[i] = top_m.theta(ti, q_m[i])
        g_N_m[i] = top_m.g_N(ti, q_m[i])

    if show_plots:
        
        # plot comparision
        fig, ax = plt.subplots(2, 1)

        ax[0].set_xlabel('t [s]')
        ax[0].set_ylabel('theta [°]')
        ax[0].plot(t_g, theta_g * 180 / pi, '-k', label='gen-al')
        ax[0].plot(t_m, theta_m * 180 / pi, '-r', label='moreau')
        ax[0].legend()

        ax[1].set_xlabel('t [s]')
        ax[1].set_ylabel('g_Ni [m]')
        ax[1].plot(t_g, g_N_g[:, 0], '-k', label='g_N1_g')
        ax[1].plot(t_g, g_N_g[:, 1], '--k', label='g_N2_g')
        ax[1].plot(t_m, g_N_m[:, 0], '-r', label='g_N1_m')
        ax[1].plot(t_m, g_N_m[:, 1], '--r', label='g_N2_m')
        ax[1].legend()

        plt.show()
    
    