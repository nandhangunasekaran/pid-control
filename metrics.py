"""
The four step-response measures (Section 4).

rise time        : time to go from 10% to 90% of the final value
overshoot        : how far the peak goes past the final value, % of the final value
settling time    : time after which the response stays within +/-2% of the final value
(Measured against the final value -- the standard textbook definition -- so the measures
still make sense for P-only control, which never reaches the setpoint.)
steady-state err : setpoint minus final value (mean of the last 5% of the run)
"""

import numpy as np


def step_metrics(t, y, setpoint=1.0, band=0.02, t_step=0.0):
    mask = t >= t_step
    t, y = t[mask] - t_step, y[mask]

    final = np.mean(y[int(0.95 * len(y)):])
    ess = setpoint - final

    ref = final if abs(final) > 1e-9 else setpoint
    # Rise time 10% -> 90% of the final value
    def first_cross(level):
        idx = np.where(y >= level)[0]
        return t[idx[0]] if len(idx) else np.nan
    tr = first_cross(0.9 * ref) - first_cross(0.1 * ref)

    peak = np.max(y)
    overshoot = max(0.0, (peak - ref) / ref * 100.0)
    t_peak = t[np.argmax(y)]

    # Settling: last time it was outside +/-2% of the final value
    outside = np.where(np.abs(y - ref) > band * abs(ref))[0]
    if len(outside) == 0:
        ts = 0.0
    elif outside[-1] == len(y) - 1:
        ts = np.nan                       # never settles inside the band (e.g. P-only offset)
    else:
        ts = t[outside[-1] + 1]

    return dict(rise_time=tr, overshoot=overshoot, settling_time=ts,
                ss_error=ess, final_value=final, peak=peak, t_peak=t_peak)


def fmt(m):
    def f(x, unit="", nd=3):
        return "never" if np.isnan(x) else f"{x:.{nd}f}{unit}"
    return (f"rise {f(m['rise_time'], ' s')}, overshoot {m['overshoot']:.1f}%, "
            f"settle(2%) {f(m['settling_time'], ' s')}, ss error {m['ss_error']*1000:.1f} mm")
