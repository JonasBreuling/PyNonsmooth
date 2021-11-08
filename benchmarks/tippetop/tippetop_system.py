# This file implements the tippetop system
#
# Stuttgart, September 2021                      G.Capobianco

import numpy as np
from numpy.linalg import norm
from math import acos, sin, cos, pi

def A_IK_basic_x(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[1,  0,   0],\
                     [0, cp, -sp],\
                     [0, sp,  cp]])

def dA_IK_basic_x(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[0,  0,   0],\
                     [0, -sp, -cp],\
                     [0, cp,  -sp]])

def A_IK_basic_y(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ cp,  0,  sp],\
                     [  0,  1,   0],\
                     [-sp,  0,  cp]])

def dA_IK_basic_y(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ -sp,  0,  cp],\
                     [  0,  0,   0],\
                     [-cp,  0,  -sp]])

def A_IK_basic_z(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ cp, -sp, 0],\
                     [ sp,  cp, 0],\
                     [  0,   0, 1]])

def dA_IK_basic_z(phi):
    sp = sin(phi)
    cp = cos(phi)
    return np.array([[ -sp, -cp, 0],\
                     [ cp,  -sp, 0],\
                     [  0,   0, 0]])

def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], \
                     a[2] * b[0] - a[0] * b[2], \
                     a[0] * b[1] - a[1] * b[0] ])

def ax2skew(a):
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]])

def ax2skew_a():
    A = np.zeros((3, 3, 3))
    A[1, 2, 0] = -1
    A[2, 1, 0] =  1
    A[0, 2, 1] =  1
    A[2, 0, 1] = -1
    A[0, 1, 2] = -1
    A[1, 0, 2] =  1
    return A

def quat2rot(p):
    v_p_tilde = ax2skew(p[1:])
    return np.eye(3) + 2 * (v_p_tilde @ v_p_tilde  + p[0] * v_p_tilde)

def quat2rot_p(p):
    v_p_tilde = ax2skew(p[1:])
    v_p_tilde_v_p = ax2skew_a()
    
    A_p = np.zeros((3, 3, 4))
    A_p[:, :, 0] = 2 * v_p_tilde
    A_p[:, :, 1:] += np.einsum('ijk,jl->ilk', v_p_tilde_v_p, 2 * v_p_tilde)
    A_p[:, :, 1:] += np.einsum('ij,jkl->ikl', 2 * v_p_tilde, v_p_tilde_v_p)
    A_p[:, :, 1:] += 2 * (p[0] * v_p_tilde_v_p)
    
    return A_p

def axis_angle2quat(axis, angle):
    return np.concatenate([ [np.cos(angle/2)], np.sin(angle/2)*axis])

class Tippetop_Leine2003():
    def __init__(self, m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0):
        """Leine2003, Chapter 5"""
        
        # Dynamics:
        self.m = m
        self.I1 = I1
        self.I3 = I3
        self.g_ = g
        # Geometry:
        self.a1 = a1
        self.a2 = a2
        self.r1 = r1
        self.r2 = r2

        # Contact:
        self.Ri_bar = 3 * pi / 16 * R
        self.Ai = np.diag([1, 1, self.Ri_bar, 1, 1, self.Ri_bar])

        self.nq = 6
        self.nu = 6
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_N = 2
        self.nla_F = 6

        self.mu = mu * np.ones(self.nla_N)
        self.e_N = eN * np.ones(self.nla_N)
        self.e_F = eF * np.ones(self.nla_F)

        self.prox_r_N = prox_r * np.ones(self.nla_N)
        self.prox_r_F = prox_r * np.ones(self.nla_N)

        self.NF_connectivity = [[0, 1, 2],
                                [3, 4, 5]]

        # initial conditions:
        self.q0 = np.zeros(self.nq)
        self.q0[2] = z0
        self.q0[3] = theta0

        self.u0 = np.zeros(self.nu)
        self.u0[5] = psi_dot0

        self.la_g0 = np.zeros(self.nla_g)
        self.la_gamma0 = np.zeros(self.nla_gamma)

        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def theta(self, t, q):
        return q[3]

    def r_OS(self, t, q):
        return q[:3]

    def A_IK(self, t, q):
        theta, phi, psi = q[3:]
        A_12 = A_IK_basic_z(phi)
        A_23 = A_IK_basic_x(theta)
        A_34 = A_IK_basic_z(psi)
        return A_12 @ A_23 @ A_34

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        x, y, z, theta, phi, psi = q
        M = np.zeros((6, 6))
        M[0, 0] = self.m
        M[1, 1] = self.m
        M[2, 2] = self.m
        M[3, 3] = self.I1
        M[4, 4] = self.I1 * sin(theta)**2 + self.I3 * cos(theta)**2
        M[4, 5] = self.I3 * cos(theta)
        M[5, 4] = self.I3 * cos(theta)
        M[5, 5] = self.I3
        return M

    def h(self, t, q, u):
        x, y, z, theta, phi, psi = q
        x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot = u
        return np.array([0,
                         0,
                         -self.m * self.g_,
                         ((self.I1 - self.I3) * phi_dot * cos(theta) - self.I3 * psi_dot) * phi_dot * sin(theta),
                         - (2 * (self.I1 - self.I3) * phi_dot * cos(theta) - self.I3 * psi_dot) * theta_dot * sin(theta),
                         self.I3 * phi_dot * theta_dot * sin(theta)])

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, a):
        return a

    def B(self, t, q):
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
        x, y, z, theta, phi, psi = q
        return np.array([z + self.a1 * cos(theta) - self.r1,
                         z + self.a2 * cos(theta) - self.r2])

    def W_N(self, t, q):
        x, y, z, theta, phi, psi = q
        g_N_q = np.zeros((2, 6))
        g_N_q[0, 2] = 1
        g_N_q[0, 3] = - self.a1 * sin(theta)
        g_N_q[1, 2] = 1
        g_N_q[1, 3] = - self.a2 * sin(theta)
        return g_N_q.T

    def g_N_dot(self, t, q, u):
        x, y, z, theta, phi, psi = q
        x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot = u
        return np.array([z_dot - self.a1 * theta_dot * sin(theta),
                         z_dot - self.a2 * theta_dot * sin(theta)])
    
    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def g_N_ddot(self, t, q, u, u_dot):
        x, y, z, theta, phi, psi = q
        x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot = u
        x_ddot, y_ddot, z_ddot, theta_ddot, phi_ddot, psi_ddot = u_dot
        fac = (theta_ddot * sin(theta) - theta_dot**2 * cos(theta))
        return np.array([z_ddot - self.a1 * fac,
                         z_ddot - self.a2 * fac])

    #################
    # friction
    #################
    def gamma_F(self, t, q, u):
        x, y, z, theta, phi, psi = q
        x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot = u
        c1 = (self.a1 * phi_dot + self.r1 * psi_dot) * sin(theta)
        c2 = theta_dot * (self.a1 * cos(theta) - self.r1)
        c3 = (self.a2 * phi_dot + self.r2 * psi_dot) * sin(theta)
        c4 = theta_dot * (self.a2 * cos(theta) - self.r2)
        ga_F =  np.array([x_dot + c1 * cos(phi) + c2 * sin(phi),
                          y_dot + c1 * sin(phi) - c2 * cos(phi),
                          (phi_dot + psi_dot * cos(theta)) ,
                          x_dot + c3 * cos(phi) + c4 * sin(phi),
                          y_dot + c3 * sin(phi) - c4 * cos(phi),
                          (phi_dot + psi_dot * cos(theta)) ])

        # print(f'L: {ga_F}')
        return self.Ai.T @ ga_F

    def gamma_F_u(self, t, q):
        x, y, z, theta, phi, psi = q
        c1 = self.a1 * cos(theta) - self.r1
        c2 = self.a2 * cos(theta) - self.r2
        dense = np.zeros((6, 6))
        dense[0, 0] = 1
        dense[0, 3] = c1 * sin(phi)
        dense[0, 4] = self.a1 * sin(theta) * cos(phi)
        dense[0, 5] = self.r1 * sin(theta) * cos(phi)
        dense[1, 1] = 1
        dense[1, 3] = - c1 * cos(phi)
        dense[1, 4] = self.a1 * sin(theta) * sin(phi)
        dense[1, 5] = self.r1 * sin(theta) * sin(phi)
        dense[2, 4] = 1
        dense[2, 5] =  cos(theta)

        dense[3, 0] = 1
        dense[3, 3] = c2 * sin(phi)
        dense[3, 4] = self.a2 * sin(theta) * cos(phi)
        dense[3, 5] = self.r2 * sin(theta) * cos(phi)
        dense[4, 1] = 1
        dense[4, 3] = - c2 * cos(phi)
        dense[4, 4] = self.a2 * sin(theta) * sin(phi)
        dense[4, 5] = self.r2 * sin(theta) * sin(phi)
        dense[5, 4] = 1
        dense[5, 5] = cos(theta)
        return self.Ai.T @ dense

    def W_F(self, t, q):
        return self.gamma_F_u(t, q).T

    def gamma_F_u_dot(self, t, q, u):
        x, y, z, theta, phi, psi = q
        x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot = u
        c1 = self.a1 * cos(theta) - self.r1
        c2 = self.a2 * cos(theta) - self.r2
        c1_dot = - self.a1 * sin(theta) * theta_dot
        c2_dot = - self.a2 * sin(theta) * theta_dot
        dense = np.zeros((6, 6))

        dense[0, 3] = c1_dot * sin(phi) + c1 * cos(phi) * phi_dot
        dense[0, 4] = self.a1 * (cos(theta) * cos(phi) * theta_dot - sin(theta) * sin(phi) * phi_dot)
        dense[0, 5] = self.r1 * (cos(theta) * cos(phi) * theta_dot - sin(theta) * sin(phi) * phi_dot)
        dense[1, 3] = - c1_dot * cos(phi) + c1 * sin(phi) * phi_dot
        dense[1, 4] = self.a1 * (cos(theta) * sin(phi) * theta_dot + sin(theta) * cos(phi) * phi_dot)
        dense[1, 5] = self.r1 * (cos(theta) * sin(phi) * theta_dot + sin(theta) * cos(phi) * phi_dot)
        dense[2, 5] = - sin(theta) * theta_dot

        dense[3, 3] = c2_dot * sin(phi) + c2 * cos(phi) * phi_dot
        dense[3, 4] = self.a2 * (cos(theta) * cos(phi) * theta_dot - sin(theta) * sin(phi) * phi_dot)
        dense[3, 5] = self.r2 * (cos(theta) * cos(phi) * theta_dot - sin(theta) * sin(phi) * phi_dot)
        dense[4, 3] = - c2_dot * cos(phi) + c2 * sin(phi) * phi_dot
        dense[4, 4] = self.a2 * (cos(theta) * sin(phi) * theta_dot + sin(theta) * cos(phi) * phi_dot)
        dense[4, 5] = self.r2 * (cos(theta) * sin(phi) * theta_dot + sin(theta) * cos(phi) * phi_dot)
        dense[5, 5] = - sin(theta) * theta_dot
        return self.Ai.T @ dense

    def gamma_F_dot(self, t, q, u, a):
        return self.gamma_F_u(t, q) @ a + self.gamma_F_u_dot(t, q, u) @ u

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)
    
class Tippetop_quaternion():
    def __init__(self, m, I1, I3, g, a1, a2, r1, r2, R, eN, eF, mu, prox_r, z0, theta0, psi_dot0):
        """Tippe top with unit-quaternions as parametrization for the orientation"""        
        # # Dynamics:
        self.m = m
        self.I1 = I1
        self.I3 = I3
        self.Theta_S = np.diag([I1, I1, I3])
        self.g_ = g
        # Geometry:
        self.a1 = a1
        self.a2 = a2
        self.r1 = r1
        self.r2 = r2
        self.K_r_SC1 = np.array([0, 0, a1])
        self.K_r_SC2 = np.array([0, 0, a2])

        # Contact:
        self.Ri_bar = 3 * pi / 16 * R
        self.Ai = np.diag([1, 1, self.Ri_bar, 1, 1, self.Ri_bar])

        self.nq = 7
        self.nu = 6
        self.nla_g = 0
        self.nla_gamma = 0
        self.nla_N = 2
        self.nla_F = 6

        self.mu = mu * np.ones(self.nla_N)
        self.e_N = eN * np.ones(self.nla_N)
        self.e_F = eF * np.ones(self.nla_F)

        self.prox_r_N = prox_r * np.ones(self.nla_N)
        self.prox_r_F = prox_r * np.ones(self.nla_N)

        self.NF_connectivity = [[0, 1, 2],
                                [3, 4, 5]]

        # initial conditions:
        self.q0 = np.zeros(self.nq)
        self.q0[2] = z0
        self.q0[3:] = axis_angle2quat(np.array([1, 0, 0]), theta0)

        self.u0 = np.zeros(self.nu)
        self.u0[5] = psi_dot0

        self.la_g0 = np.zeros(self.nla_g)
        self.la_gamma0 = np.zeros(self.nla_gamma)

        self.la_N0 = np.zeros(self.nla_N)
        self.la_F0 = np.zeros(self.nla_F)

    def theta(self, t, q):
        return acos(self.A_IK(t, q)[2, 2])

    def A_IK(self, t, q):
        return quat2rot(q[3:])

    def A_IK_q(self, t, q):
        A_IK_q = np.zeros((3, 3, self.nq))
        A_IK_q[:, :, 3:] = quat2rot_p(q[3:])
        return A_IK_q

    def r_OP(self, t, q, K_r_SP=np.zeros(3)):
        return q[:3] + self.A_IK(t, q) @ K_r_SP

    def r_OS(self, t, q):
        return self.r_OP(t, q)

    def r_OP_q(self, t, q, K_r_SP=np.zeros(3)):
        r_OP_q = np.zeros((3, self.nq))
        r_OP_q[:, :3] = np.eye(3)
        r_OP_q[:, :] += np.einsum('ijk,j->ik', self.A_IK_q(t, q), K_r_SP)
        return r_OP_q

    def v_P(self, t, q, u, K_r_SP=np.zeros(3)):
        return u[:3] + self.A_IK(t, q) @ cross(u[3:], K_r_SP)

    def a_P(self, t, q, u, u_dot, K_r_SP=np.zeros(3)):
        return u_dot[:3] + self.A_IK(t, q) @ (cross(u_dot[3:], K_r_SP) + cross(u[3:], cross(u[3:], K_r_SP)))
    
    def kappa_P(self, t, q, u, K_r_SP=np.zeros(3)):
        return self.A_IK(t, q) @ (cross(u[3:], cross(u[3:], K_r_SP)))
    
    def kappa_P_q(self, t, q, u, K_r_SP=np.zeros(3)):
        return np.einsum('ijk,j->ik', self.A_IK_q(t, q), cross(u[3:], cross(u[3:], K_r_SP)) )
    
    def kappa_P_u(self, t, q, u, K_r_SP=np.zeros(3)):
        kappa_P_u = np.zeros((3, self.nu))
        kappa_P_u[:, 3:] = -self.A_IK(t, q) @ (ax2skew(cross(u[3:], K_r_SP)) + ax2skew(u[3:]) @ ax2skew(K_r_SP))
        return kappa_P_u

    def v_P_u(self, t, q, K_r_SP=np.zeros(3)):
        J_P = np.zeros((3, self.nu))
        J_P[:, :3] = np.eye(3)
        J_P[:, 3:] = - self.A_IK(t, q) @ ax2skew(K_r_SP)
        return J_P

    def v_P_uq(self, t, q, K_r_SP=np.zeros(3)):
        J_P_q = np.zeros((3, self.nu, self.nq))
        J_P_q[:, 3:, :] = np.einsum('ijk,jl->ilk', self.A_IK_q(t, q), -ax2skew(K_r_SP))
        return J_P_q

    def K_Omega(self, t, q, u):
        return u[3:]

    def K_Psi(self, t, q, u, u_dot):
        return u_dot[3:]

    def K_Omega_u(self, t, q):
        J_R = np.zeros((3, self.nu))
        J_R[:, 3:] = np.eye(3)
        return J_R

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.diag([self.m, self.m, self.m, self.I1, self.I1, self.I3])

    def h(self, t, q, u):
        f_gravity = np.array([0,
                         0,
                         -self.m * self.g_])
        omega = u[3:]
        f_gyroscopic = - cross(omega, self.Theta_S @ omega)
        return np.concatenate([f_gravity, f_gyroscopic])

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return self.B(t, q) @ u

    def q_ddot(self, t, q, u, a):
        return self.B(t, q) @ a + self.B_dot(t, q, u) @ u

    def B(self, t, q):
        p0, p1, p2, p3 = q[3:]  
        B = np.eye(self.nq, self.nu)

        B[3:, 3:] = 0.5 * np.array([[-p1, -p2, -p3],
                                    [ p0, -p3,  p2],
                                    [ p3,  p0, -p1],
                                    [-p2,  p1,  p0]])
        return B
    
    def B_dot(self, t, q, u):
        p0_dot, p1_dot, p2_dot, p3_dot = self.q_dot(t, q, u)[3:]
        B = np.zeros((self.nq, self.nu))

        B[3:, 3:] = 0.5 * np.array([[-p1_dot, -p2_dot, -p3_dot],
                                    [ p0_dot, -p3_dot,  p2_dot],
                                    [ p3_dot,  p0_dot, -p1_dot],
                                    [-p2_dot,  p1_dot,  p0_dot]])
        return B

    def step_callback(self, q, u):
        p = q[3:]
        q[3:] = p / norm(p)
        return q, u

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
        r_OC1 = self.r_OP(t, q, K_r_SP=self.K_r_SC1)
        r_OC2 = self.r_OP(t, q, K_r_SP=self.K_r_SC2)
        # return np.array([1, 1])
        return np.array([r_OC1[2] - self.r1,
                         r_OC2[2] - self.r2])

    def W_N(self, t, q):
        v_C1_u = self.v_P_u(t, q, K_r_SP=self.K_r_SC1)
        v_C2_u = self.v_P_u(t, q, K_r_SP=self.K_r_SC2)

        W_N = np.zeros((self.nu, 2))
        W_N[:, 0] = v_C1_u[2]
        W_N[:, 1] = v_C2_u[2]
        return W_N

    def g_N_dot(self, t, q, u):
        v_C1 = self.v_P(t, q, u, K_r_SP=self.K_r_SC1)
        v_C2 = self.v_P(t, q, u, K_r_SP=self.K_r_SC2)
        return np.array([v_C1[2],
                         v_C2[2]])
    
    def xi_N(self, t, q, u_pre, u_post):
        return self.g_N_dot(t, q, u_post) + self.e_N * self.g_N_dot(t, q, u_pre)

    def g_N_ddot(self, t, q, u, u_dot):
        a_C1 = self.a_P(t, q, u, u_dot, self.K_r_SC1)
        a_C2 = self.a_P(t, q, u, u_dot, self.K_r_SC2)
        return np.array([a_C1[2],
                         a_C2[2]])
        
    #################
    # friction
    #################
    def gamma_F(self, t, q, u):
        A_IK = self.A_IK(t, q)
        K_r_SP1 = self.K_r_SC1 + A_IK.T @ np.array([0, 0, -self.r1])
        K_r_SP2 = self.K_r_SC2 + A_IK.T @ np.array([0, 0, -self.r2])
        v_C1 = self.v_P(t, q, u, K_r_SP=K_r_SP1)
        v_C2 = self.v_P(t, q, u, K_r_SP=K_r_SP2)
        Omega = A_IK @ self.K_Omega(t, q, u)

        ga_F = np.zeros(6)
        ga_F[:2] = v_C1[:2]
        ga_F[2] = Omega[2]
        ga_F[3:5] = v_C2[:2]
        ga_F[5] = Omega[2]
        return self.Ai @ ga_F

    def W_F(self, t, q):
        A_IK = self.A_IK(t, q)
        K_r_SP1 = self.K_r_SC1 + A_IK.T @ np.array([0, 0, -self.r1])
        K_r_SP2 = self.K_r_SC2 + A_IK.T @ np.array([0, 0, -self.r2])
        v_C1_u = self.v_P_u(t, q, K_r_SP1)
        v_C2_u = self.v_P_u(t, q, K_r_SP2)
        Omega_u = A_IK @ self.K_Omega_u(t, q)

        W_F = np.zeros((self.nu, 6))
        W_F[:, :2] = v_C1_u[:2].T
        W_F[:, 2] = Omega_u[2]
        W_F[:, 3:5] = v_C2_u[:2].T
        W_F[:, 5] = Omega_u[2]
        return W_F @ self.Ai

    def gamma_F_dot(self, t, q, u, a):
        A_IK = self.A_IK(t, q)
        K_r_SP1 = self.K_r_SC1 + A_IK.T @ np.array([0, 0, -self.r1])
        K_r_SP2 = self.K_r_SC2 + A_IK.T @ np.array([0, 0, -self.r2])
        a_C1 = self.a_P(t, q, u, a, K_r_SP1)
        a_C2 = self.a_P(t, q, u, a, K_r_SP2)
        Psi = A_IK @ self.K_Psi(t, q, u, a)

        ga_F_dot = np.zeros(6)
        ga_F_dot[:2] = a_C1[:2]
        ga_F_dot[2] = Psi[2]
        ga_F_dot[3:5] = a_C2[:2]
        ga_F_dot[5] = Psi[2]

        return self.Ai @ ga_F_dot

    def xi_F(self, t, q, u_pre, u_post):
        return self.gamma_F(t, q, u_post) + self.e_F * self.gamma_F(t, q, u_pre)
