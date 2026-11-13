"""
A discrete-time PID controller written from scratch (NumPy only, no control libraries).

    u(t) = Kp*e(t) + Ki * integral(e dt) + Kd * de/dt,    e = setpoint - measurement

Every real-world fix from Section 7 can be switched on/off so the problems can be
demonstrated *and* solved with the same class:

    output_limits      -> actuator saturation (e.g. motor can push at most 30 N)
    anti_windup        -> stop integrating while the output is saturated (clamping)
    derivative_on      -> "error" (textbook, causes derivative kick) or
                          "measurement" (kick-free)
    d_filter_tau       -> first-order low-pass filter on the D term (noise fix), seconds
"""

import numpy as np


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, dt=0.001,
                 output_limits=(None, None),
                 anti_windup=False,
                 derivative_on="error",
                 d_filter_tau=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = output_limits
        self.anti_windup = anti_windup
        if derivative_on not in ("error", "measurement"):
            raise ValueError("derivative_on must be 'error' or 'measurement'")
        self.derivative_on = derivative_on
        self.d_filter_tau = d_filter_tau
        self.reset()

    def reset(self):
        self.integral = 0.0        # stores Ki * integral(e dt)  [same units as u]
        self.prev_error = None
        self.prev_measurement = None
        self.d_term = 0.0          # filtered derivative contribution
        # last individual terms, handy for plotting
        self.p_term = self.i_term = 0.0

    def _saturate(self, u):
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)
        return u

    def update(self, setpoint, measurement):
        """Run one controller step. Call once every dt seconds. Returns the command u."""
        dt = self.dt
        error = setpoint - measurement

        # ---- P: react to the present error
        p = self.kp * error

        # ---- D: react to how fast things are changing
        if self.prev_error is None:          # first call: no history yet
            raw_d = 0.0
        elif self.derivative_on == "error":
            raw_d = self.kd * (error - self.prev_error) / dt
        else:  # derivative of measurement: same thing when setpoint is constant, no kick on steps
            raw_d = -self.kd * (measurement - self.prev_measurement) / dt

        if self.d_filter_tau > 0:            # first-order low-pass: d += a*(raw - d)
            alpha = dt / (self.d_filter_tau + dt)
            self.d_term += alpha * (raw_d - self.d_term)
        else:
            self.d_term = raw_d

        # ---- I: accumulate past error
        candidate_integral = self.integral + self.ki * error * dt
        u_unsat = p + candidate_integral + self.d_term
        u = self._saturate(u_unsat)

        if self.anti_windup and u != u_unsat:
            # Clamping: only accept the new integral if it pulls the output
            # back out of saturation (i.e. error and saturation have opposite signs).
            pushing_further = (u_unsat > u and error > 0) or (u_unsat < u and error < 0)
            if not pushing_further:
                self.integral = candidate_integral
            u = self._saturate(p + self.integral + self.d_term)
        else:
            self.integral = candidate_integral

        self.prev_error = error
        self.prev_measurement = measurement
        self.p_term, self.i_term = p, self.integral
        return u
