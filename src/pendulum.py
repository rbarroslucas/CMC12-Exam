import numpy as np

class DoubleInvertedPendulumCart:
    """
    Double inverted pendulum on a cart system.
    This class models a double inverted pendulum on a cart system and computes the linearized state-space representation.
    Attributes:
        m0 (float): Mass of the cart.
        m1 (float): Mass of the first pendulum.
        m2 (float): Mass of the second pendulum.
        L1 (float): Length of the first pendulum.
        L2 (float): Length of the second pendulum.
        g (float): Acceleration due to gravity.
    """
    def __init__(self, m0: float, m1: float, m2: float, L1: float, L2: float, g: float = 9.81):
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g  
        
        self.l1 = L1 / 2 
        self.l2 = L2 / 2
        
        self.I1 = (1/12) * m1 * L1**2
        self.I2 = (1/12) * m2 * L2**2
        
        self._compute_linearized_matrices()

    def get_linearized_system(self):
        return self.A, self.B
    
    def _compute_linearized_matrices(self):
        """
        Compute the linearized state-space matrices for the system.
    
        """
        m0, m1, m2 = self.m0, self.m1, self.m2
        L1, L2, g = self.L1, self.L2, self.g
        l1, l2 = self.l1, self.l2
        I1, I2 = self.I1, self.I2
        
        M11 = m0 + m1 + m2
        M12 = m1*l1 + m2*L1
        M13 = m2*l2
        M22 = m1*l1**2 + m2*L1**2 + I1
        M23 = m2*L1*l2
        M33 = m2*l2**2 + I2
        
        self.M0 = np.array([
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33]
        ])
        
        self.M0_inv = np.linalg.inv(self.M0)
        
        self.dh_dtheta = np.array([
            [0, 0, 0],
            [0, (m1*l1 + m2*L1)*g, 0],
            [0, 0, m2*l2*g]
        ])
        
        self.F = np.array([[1], [0], [0]])
        
        self._compute_state_space_matrices()
    
    def _compute_state_space_matrices(self):
        """
        Compute the state-space matrices for the system.
        """
        A21 = -self.M0_inv @ self.dh_dtheta
        B21 = self.M0_inv @ self.F
        
        self.A = np.zeros((6, 6))

        self.A[0, 3] = 1
        self.A[1, 4] = 1
        self.A[2, 5] = 1
        
        self.A[3:6, 0:3] = A21
        
        self.B = np.zeros((6, 1))
        self.B[3:6, 0] = B21.flatten()