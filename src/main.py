import warnings
import numpy as np

from mpc import LMPC
from config import config
from pendulum import DoubleInvertedPendulumCart
from utils import plot_results, animate_pendulum, plot_performance_metrics

system = DoubleInvertedPendulumCart(config.m0, config.m1, config.m2, config.L1, config.L2)

A, B = system.get_linearized_system()
A_d = np.eye(A.shape[0]) + A * config.dt
B_d = B * config.dt

controller = LMPC(A=A_d, B=B_d, N=config.N, Q=config.Q, R=config.R, Qn=config.Qn, x_min=config.x_min, x_max=config.x_max, u_min=config.u_min, u_max=config.u_max)

X_hist, U_hist, t_hist = controller.simulate(config.x0, config.N_sim, x_ref=config.x_ref, dt=config.dt)

plot_results(t_hist, X_hist, U_hist, system, config.x_ref)
animate_pendulum(system, X_hist, "pendulum_mpc_simulation.gif")
plot_performance_metrics(t_hist, X_hist, config.x_ref)

print(f"Simulation completed: {config.N_sim} steps")