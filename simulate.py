"""
Closed-loop simulation.

The physics runs on a fine time step (dt_sim, default 0.1 ms) while the controller only
runs every controller.dt seconds and holds its output in between (zero-order hold) --
exactly what a real microcontroller does. This is what lets Section 7(d) show the effect
of a slow update rate.
"""

import numpy as np
from plants import rk4_step


def step_setpoint(value=1.0, t_step=0.0):
    return lambda t: value if t >= t_step else 0.0


def simulate(plant, controller, setpoint_fn, t_end, dt_sim=1e-4,
             noise_amp=0.0, seed=0):
    """
    noise_amp: sensor noise, uniform in [-noise_amp, +noise_amp] (metres for the block).
    Returns dict of arrays: t, y (true output), y_meas, sp, u, p, i, d
    """
    rng = np.random.default_rng(seed)
    n_sub = max(1, int(round(controller.dt / dt_sim)))
    h = controller.dt / n_sub
    n_ctrl = int(round(t_end / controller.dt))

    log = {k: [] for k in ("t", "y", "y_meas", "sp", "u", "p", "i", "d")}
    t = 0.0
    for _ in range(n_ctrl):
        y = plant.output()
        y_meas = y + (rng.uniform(-noise_amp, noise_amp) if noise_amp > 0 else 0.0)
        sp = setpoint_fn(t)
        u = controller.update(sp, y_meas)
        # log at controller rate, then hold u while physics runs
        for key, val in zip(log, (t, y, y_meas, sp, u, controller.p_term,
                                  controller.i_term, controller.d_term)):
            log[key].append(val)
        for _ in range(n_sub):
            rk4_step(plant, u, h)
        t += controller.dt
    return {k: np.asarray(v) for k, v in log.items()}
