import numpy as np
import cvxpy as cp

class LMPC:
    def __init__(self, A, B, N, Q, R, Qn=None, x_min=None, x_max=None, u_min=None, u_max=None):
        self.A = A
        self.B = B
        self.N = N
        self.n = A.shape[0] 
        self.m = B.shape[1]
        self.Q = Q
        self.R = R
        self.Qn = Qn if Qn is not None else Q
        
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        
        self.x_ref = np.zeros(self.n)
        self.u_ref = np.zeros(self.m)
        
    def solve_mpc(self, x_current):
        X = cp.Variable((self.N + 1, self.n))
        U = cp.Variable((self.N, self.m))
        
        constraints = []
        constraints.append(X[0] == x_current)

        for k in range(self.N):
            constraints.append(X[k + 1] == self.A @ X[k] + self.B @ U[k])
        
        if self.x_min is not None:
            for k in range(self.N + 1):
                constraints.append(X[k] >= self.x_min)
        if self.x_max is not None:
            for k in range(self.N + 1):
                constraints.append(X[k] <= self.x_max)
        
        if self.u_min is not None:
            for k in range(self.N):
                constraints.append(U[k] >= self.u_min)
        if self.u_max is not None:
            for k in range(self.N):
                constraints.append(U[k] <= self.u_max)
        
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(X[k] - self.x_ref, self.Q)
            cost += cp.quad_form(U[k] - self.u_ref, self.R)
        
        cost += cp.quad_form(X[self.N] - self.x_ref, self.Qn)
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status in ["infeasible", "unbounded"]:
            print(f"MPC problem status: {problem.status}")
            return np.zeros(self.m), None, None
        
        return U.value[0], X.value, U.value

    def set_reference(self, x_ref):
        self.x_ref = x_ref
        self.u_ref = np.zeros(self.m)
    
    def simulate(self, x0, N_sim, x_ref=None, dt=0.1):
        if x_ref is not None:
            self.set_reference(x_ref)
        
        X_hist = np.zeros((N_sim + 1, self.n))
        U_hist = np.zeros((N_sim, self.m))
        t_hist = np.linspace(0, N_sim * dt, N_sim + 1)
        
        x = x0.copy()
        X_hist[0] = x
        
        for k in range(N_sim):
            u_opt, X_opt, U_opt = self.solve_mpc(x)
            
            x = self.A @ x + self.B @ u_opt
            
            X_hist[k + 1] = x
            U_hist[k] = u_opt
            
            print(f"Step {k}/{N_sim}, State: {x[:3]}, Control: {u_opt}")
        
        return X_hist, U_hist, t_hist