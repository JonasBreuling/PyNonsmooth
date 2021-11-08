#----------------------------------------------------------------------
# A nonsmooth generalized-alpha method for mechanical systems
# with frictional contact
# 
# Giuseppe Capobianco, Jonas Harsch, Simon R. Eugster, Remco I. Leine
#----------------------------------------------------------------------
#Int J Numer Methods Eng. 2021; 1– 30. https://doi.org/10.1002/nme.6801
#----------------------------------------------------------------------
# 
# This file implements Moreau's time-stepping scheme as described in
#
# G. Capobianco and S.R. Eugster: Time finite element based Moreau-type
# integrators, International Journal for Numerical Methods in 
# Engineering, Vol. 114(3), pp. 215-345, 2018.
# https://doi.org/10.1002/nme.5741
#
# Stuttgart, September 2021                     G.Capobianco, J. Harsch

import numpy as np
from numpy.linalg import norm, solve
from tqdm import tqdm

# proximal point function to the set of negative numbers including zero
def prox_Rn0(x):
    return np.maximum(x, 0)

# proximal point function to a sphere 
def prox_sphere(x, radius):
    nx = norm(x)
    if nx > 0:
        return x if nx <= radius else radius * x / nx
    else:
        return x if nx <= radius else radius * x

class Moreau():
    def __init__(self, model, t0, t1, dt, fix_point_tol=1e-8, fix_point_max_iter=1000, error_function=lambda x: np.max(np.abs(x))):
        self.model = model

        # integration time
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.fix_point_error_function = error_function
        self.fix_point_tol = fix_point_tol
        self.fix_point_max_iter = fix_point_max_iter

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.nla_N = self.model.nla_N
        self.nla_F = self.model.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        self.tk = t0
        self.qk = model.q0 
        self.uk = model.u0 
        self.la_gk = model.la_g0
        self.la_gammak = model.la_gamma0
        self.P_Nk = model.la_N0 * dt
        self.P_Fk = model.la_F0 * dt

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = self.model.NF_connectivity

        self.DOFs_smooth = np.arange(self.nR_smooth)

        if hasattr(model, 'step_callback'):
            self.step_callback = model.step_callback
        else:
            self.step_callback = self.__step_callback
    
    def __step_callback(self, q, u):
        return q, u

    def step(self):
        # general quantities
        dt = self.dt
        uk = self.uk
        tk1 = self.tk + dt

        # position update
        qk1 = self.qk + dt * self.model.q_dot(self.tk, self.qk, uk)

        # get quantities from model
        M = self.model.M(tk1, qk1)
        h = self.model.h(tk1, qk1, uk)
        W_N = self.model.W_N(tk1, qk1)
        W_F = self.model.W_F(tk1, qk1)

        # identify active normal and tangential contacts
        g_N = self.model.g_N(tk1, qk1)
        I_N = (g_N <= 0)
        if np.any(I_N):
            I_F = np.array([c for i, I_N_i in enumerate(I_N) for c in self.model.NF_connectivity[i] if I_N_i], dtype=int)
        else:
            I_F = np.array([], dtype=int)

        # initialize quantities for fixed-point iterations
        la_gk1 = self.la_gk
        la_gammak1 = self.la_gammak
        uk1 =  uk + solve(M, dt * h + W_N[:, I_N] @ self.P_Nk[I_N] + W_F[:, I_F] @ self.P_Fk[I_F])
        P_Nk1 = np.zeros(self.nla_N)
        P_Fk1 = np.zeros(self.nla_F)

        converged = True
        error = 0
        j = 0
        # if any contact is active
        if np.any(I_N):
            converged = False
            P_Nk1_i = self.P_Nk.copy()
            P_Nk1_i1 = self.P_Nk.copy()
            P_Fk1_i = self.P_Fk.copy()
            P_Fk1_i1 = self.P_Fk.copy()
            # fixed-point iterations
            for j in range(self.fix_point_max_iter):
                
                # fixed-point update normal direction
                P_Nk1_i1[I_N] = prox_Rn0(P_Nk1_i[I_N] - self.model.prox_r_N[I_N] * self.model.xi_N(tk1, qk1, uk, uk1)[I_N])

                # fixed-point update friction
                xi_F = self.model.xi_F(tk1, qk1, uk, uk1)
                for i_N, i_F in enumerate(self.NF_connectivity):
                    if I_N[i_N] and len(i_F):
                        P_Fk1_i1[i_F] = prox_sphere(P_Fk1_i[i_F] - self.model.prox_r_F[i_N] * xi_F[i_F], self.model.mu[i_N] * P_Nk1_i1[i_N]) 

                # check for convergence
                # error = self.fix_point_error_function(uk1 - uk0)
                R = np.concatenate( (P_Nk1_i1[I_N] - P_Nk1_i[I_N], P_Fk1_i1[I_F] - P_Fk1_i[I_F]) )
                error = self.fix_point_error_function(R)
                converged = error < self.fix_point_tol
                if converged:
                    P_Nk1[I_N] = P_Nk1_i1[I_N]
                    P_Fk1[I_F] = P_Fk1_i1[I_F]
                    break
                P_Nk1_i = P_Nk1_i1.copy()
                P_Fk1_i = P_Fk1_i1.copy()

                # velocity update
                uk1 =  uk + solve(M, dt * h + W_N[:, I_N] @ P_Nk1_i[I_N] + W_F[:, I_F] @ P_Fk1_i[I_F])

        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, P_Nk1, P_Fk1


    def solve(self):
        
        # lists storing output variables
        q = [self.qk]
        u = [self.uk]
        la_g = [self.la_gk]
        la_gamma = [self.la_gammak]
        P_N = [self.P_Nk]
        P_F = [self.P_Fk]

        pbar = tqdm(self.t[:-1])
        for _ in pbar:
            (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, P_Nk1, P_Fk1 = self.step()
            pbar.set_description(f't: {tk1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}')
            if not converged:
                raise RuntimeError(f'fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}')

            qk1, uk1 = self.step_callback(qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            P_N.append(P_Nk1)
            P_F.append(P_Fk1)

            # update local variables for accepted time step
            self.tk, self.qk, self.uk, self.la_gk, self.la_gammak, self.P_Nk, self.P_Fk = tk1, qk1, uk1, la_gk1, la_gammak1, P_Nk1, P_Fk1
            
        # write solution
        return self.t, np.array(q), np.array(u), np.array(la_g), np.array(la_gamma), np.array(P_N), np.array(P_F)
