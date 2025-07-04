import numpy as np
from types import SimpleNamespace

config = SimpleNamespace(
    m0=1.5,
    m1=0.5,
    m2=0.75,
    L1=0.5,
    L2=0.75,
    x0=np.array([0.0, 0.6, -0.6, 0.0, 0.0, 0.0]),
    N=20,
    dt=0.05,
    Q=np.diag([10, 100, 100, 1, 10, 10]),
    R=np.array([[0.1]]),
    Qn=np.diag([10, 100, 100, 1, 10, 10]) * 2,
    x_max=np.array([2, 0.5, 0.5, 3, 5, 5]),
    x_min=-np.array([2, 0.5, 0.5, 3, 5, 5]),
    u_max=np.array([50]),
    u_min=-np.array([50]),
    x_ref=np.array([0, 0, 0, 0, 0, 0]),
    N_sim=400
)