# This file implements the rotating bouncing ball example of 
# Section 10.1 in "A nonsmooth generalized-alpha method for 
# mechanical systems with frictional contact", Giuseppe Capobianco, 
# Jonas Harsch, Simon R. Eugster, Remco I. Leine
# Int J Numer Methods Eng. 2021; 1– 30. https://doi.org/10.1002/nme.6801
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
import matplotlib.pyplot as plt

from bouncing_ball_system import Bouncing_ball

from solvers.moreau import Moreau
from solvers.generalized_alpha import Generalized_alpha

if __name__ == "__main__":
    
    # case 1: Figure 1
    # case 2: Figure 2 (left)
    # case 3: Figure 2 (right)
    case = 3

    # show simulation results
    show_plots = True

    if case == 1:
        eN, eF, mu = 0.5, 0, 0.2
        q0 = np.array([0, 1, 0])
        u0 = np.array([0, 0, 0])
        t1 = 1.5
    elif case == 2:
        eN, eF, mu = 0, 0, 0.2
        q0 = np.array([0, 1, 0])
        u0 = np.array([0, 0, 50])
        t1 = 1.1
    elif case == 3:
        eN, eF, mu = 0, 0, 0.2
        q0 = np.array([0, 1, 0])
        u0 = np.array([0, 0, 10])
        t1 = 1.1
    elif case == 4:
        # more complex example for benchmarking purposes
        eN, eF, mu = 0.5, 0, 0.2
        q0 = np.array([-0.5, 1, 0])
        u0 = np.array([1, 0, 50])
        t1 = 1.5
    else:
        raise AssertionError('Case not found!')

    mass = 1
    radius = 0.1
    ball = Bouncing_ball(mass, radius, eN, eF, mu, q0, u0)

    t0 = 0 # initial simulation time
    dt = 2e-3 # time step

    # solve with nonsmooth generalized-alpha scheme
    gen_al = Generalized_alpha(ball, t0, t1, dt, rho_inf=0.5, newton_tol=1.0e-6)    
    sol_g = gen_al.solve()

    t_g = sol_g[0]
    q_g = sol_g[1]
    u_g = sol_g[2]
    a_g = sol_g[3]
    La_N_g = sol_g[10]
    la_N_g = sol_g[11]
    La_F_g = sol_g[12]
    la_F_g = sol_g[13]
    P_N_g = sol_g[14]
    P_F_g = sol_g[15]

    # solve with Moreau's midpoint rule
    moreau = Moreau(ball, t0, t1, dt, fix_point_tol=1e-8)
    sol_m = moreau.solve()
    t_m = sol_m[0]
    q_m = sol_m[1]
    u_m = sol_m[2]
    P_N_m = sol_m[5]
    P_F_m = sol_m[6]

    if show_plots:
        
        fig, ax = plt.subplots(3, 1)

        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].plot(t_g, q_g[:, 0], '-k', label='gen_alpha')
        ax[0].plot(t_m, q_m[:, 0], '--r', label='moreau')
        ax[0].legend()

        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$u_x$')
        ax[1].plot(t_g, u_g[:, 0], '-k', label='gen_alpha')
        ax[1].plot(t_m, u_m[:, 0], '--r', label='moreau')
        ax[1].legend()

        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$a_x$')
        ax[2].plot(t_g, a_g[:, 0], '-k', label='gen_alpha')
        ax[2].legend()
        
        ######
        fig, ax = plt.subplots(3, 1)

        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$y$')
        ax[0].plot(t_g, q_g[:, 1], '-k', label='gen_alpha')
        ax[0].plot(t_m, q_m[:, 1], '--r', label='moreau')
        ax[0].legend()

        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$u_y$')
        ax[1].plot(t_g, u_g[:, 1], '-k', label='gen_alpha')
        ax[1].plot(t_m, u_m[:, 1], '--r', label='moreau')
        ax[1].legend()

        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$a_y$')
        ax[2].plot(t_g, a_g[:, 1], '-k', label='gen_alpha')
        ax[2].legend()
        
        ######
        fig, ax = plt.subplots(3, 1)

        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel(r'$\varphi$')
        ax[0].plot(t_g, q_g[:, 2], '-k', label='gen_alpha')
        ax[0].plot(t_m, q_m[:, 2], '--r', label='moreau')
        ax[0].legend()

        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel(r'$u_\varphi$')
        ax[1].plot(t_g, u_g[:, 2], '-k', label='gen_alpha')
        ax[1].plot(t_m, u_m[:, 2], '--r', label='moreau')
        ax[1].legend()

        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel(r'$a_\varphi$')
        ax[2].plot(t_g, a_g[:, 2], '-k', label='gen_alpha')
        ax[2].legend()

        ######
        fig, ax = plt.subplots(2, 1)
        ax[0].set_title('force comp. by gen-alpha')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('force')
        ax[0].plot(t_g, la_N_g[:, 0], '-b', label='$\lambda_N$')
        ax[0].plot(t_g, La_N_g[:, 0], '--r', label='$\Lambda_N$')
        ax[0].legend()

        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('force')
        ax[1].plot(t_g, la_F_g[:, 0], '-b', label='$\lambda_F$')
        ax[1].plot(t_g, La_F_g[:, 0], '--r', label='$\Lambda_F$')
        ax[1].legend()

        ######
        fig, ax = plt.subplots(2, 1)

        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$P_N$')
        ax[0].plot(t_g, P_N_g[:, 0], '-k', label='gen_alpha')
        ax[0].plot(t_m, P_N_m[:, 0], '--r', label='moreau')
        ax[0].legend()

        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$P_F$')
        ax[1].plot(t_g, P_F_g[:, 0], '-k', label='gen_alpha')
        ax[1].plot(t_m, P_F_m[:, 0], '--r', label='moreau')
        ax[1].legend()

    plt.show()