# This file implements the rotating bouncing ball system, 
# see Section 10.1 in "A nonsmooth generalized-alpha method for 
# mechanical systems with frictional contact", Giuseppe Capobianco, 
# Jonas Harsch, Simon R. Eugster, Remco I. Leine
# Int J Numer Methods Eng. 2021; 1– 30. https://doi.org/10.1002/nme.6801
#
# Stuttgart, November 2021                                  G.Capobianco

import numpy as np
from math import sin, cos

# basic rotation around z-axis
def A_IK_basic_z(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ cp, -sp, 0],\
                     [ sp,  cp, 0],\
                     [  0,   0, 1]])

class Bouncing_ball():
    def __init__(self, mass, radius, eN, eF, mu, q0, u0):
        
        self.m = mass
        self.R = radius
        self.theta = 2 / 5 * self.m * self.R**2 # rotational inertia
        self.g_ = 9.81 # gravitational acceleration
        
        # set numbers of degree of freedom
        self.nq = 3
        self.nu = 3
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_N = 1
        self.nla_F = 1

        # contact parameters
        self.mu = mu * np.ones(self.nla_N)
        self.e_N = eN * np.ones(self.nla_N)
        self.e_F = eF * np.ones(self.nla_F)

        # parameter for prox-formulation of contact laws
        prox_r = 0.3
        self.prox_r_N = prox_r * np.ones(self.nla_N)
        self.prox_r_F = prox_r * np.ones(self.nla_N)

        # connectivity of friction and normal forces
        self.NF_connectivity = [[0]]
        
        # set initial conditions
        self.q0 = q0 # (x0, y0, phi0)
        self.u0 = u0 # (x_dot0, y_dot0, phi_dot0)

        self.la_g0 = np.zeros(self.nla_g)
        self.la_gamma0 = np.zeros(self.nla_gamma)

        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    # postition vector to center of gravity
    def r_OS(self, t, q):
        r_OS = np.zeros(3)
        r_OS[:2] = q[:2]
        return r_OS

    # transformation matrix from body fixed to inertial system
    def A_IK(self, t, q):
        return A_IK_basic_z(q[2])

    #####################
    # equations of motion
    #####################

    # mass matrix
    def M(self, t, q):
        return np.diag([self.m, self.m, self.theta])

    # total force vector
    def h(self, t, q, u):
        return np.array([0,
                         - self.m * self.g_,
                         0])

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, a):
        return a

    def B(self, t, q): # d(q_dot)/du
        return np.eye(self.nu)

    #######################
    # bilateral constraints
    #######################
    def g(self, t, q):
        return np.zeros(self.nla_g)

    def W_g(self, t, q):
        return np.zeros((self.nu, self.nla_g))

    def g_dot(self, t, q, u):
        return np.zeros(self.nla_g)

    def g_ddot(self, t, q, u, a):
        return np.zeros(self.nla_g)

    def gamma(self, t, q, u):
        return np.zeros(self.nla_gamma)

    def W_gamma(self, t, q):
        return np.zeros((self.nu, self.nla_gamma))

    def gamma_dot(self, t, q, u, a):
        return np.zeros(self.nla_gamma)

    #################
    # normal contacts
    #################
    def g_N(self, t, q):
        return np.array([q[1] - self.R])

    def W_N(self, t, q):
        W_N = np.zeros((self.nu, self.nla_N))
        W_N[1 , 0] = 1
        return W_N

    def g_N_dot(self, t, q, u):
        return np.array([u[1]])
    
    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def g_N_ddot(self, t, q, u, a):
        return np.array([ a[1] ])

    #################
    # friction
    #################
    def gamma_F(self, t, q, u):
        x_dot, y_dot, phi_dot = u
        return np.array([ x_dot + self.R * phi_dot])

    def gamma_F_u(self, t, q):
        gamma_F_u = np.zeros((self.nla_N, self.nu))
        gamma_F_u[0 , 0] = 1
        gamma_F_u[0 , 2] = self.R
        return gamma_F_u

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def gamma_F_dot(self, t, q, u, a):
        x_ddot, y_ddot, phi_ddot = a
        return np.array([ x_ddot + self.R * phi_ddot])

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)
    