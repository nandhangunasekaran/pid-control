"""
Plants (the machines being controlled), written as state-space ODEs and integrated with RK4.

MassSpringDamper:  m*x'' + c*x' + k*x = F        (Section 3 test system)
ThirdOrderLag:     G(s) = 1 / (s + 1)^3          (Section 8 Ziegler-Nichols system)
"""

import numpy as np


class MassSpringDamper:
    """State = [position x (m), velocity v (m/s)], input = force F (N)."""

    def __init__(self, m=1.0, c=10.0, k=20.0, x0=0.0, v0=0.0):
        self.m, self.c, self.k = m, c, k
        self.state = np.array([x0, v0], dtype=float)

    def deriv(self, s, F):
        x, v = s
        a = (F - self.c * v - self.k * x) / self.m
        return np.array([v, a])

    def output(self):
        return self.state[0]


class ThirdOrderLag:
    """1/(s+1)^3 built as three first-order lags in series: x1'=-x1+u, x2'=-x2+x1, x3'=-x3+x2."""

    def __init__(self):
        self.state = np.zeros(3)

    def deriv(self, s, u):
        x1, x2, x3 = s
        return np.array([-x1 + u, -x2 + x1, -x3 + x2])

    def output(self):
        return self.state[2]


def rk4_step(plant, u, h):
    s = plant.state
    k1 = plant.deriv(s, u)
    k2 = plant.deriv(s + 0.5 * h * k1, u)
    k3 = plant.deriv(s + 0.5 * h * k2, u)
    k4 = plant.deriv(s + h * k3, u)
    plant.state = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
