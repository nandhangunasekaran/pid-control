# PID Control from Scratch

The code behind my PID control report for the Modern Monk Labs internship assignment.
The PID logic, plant models, simulation loop and metrics are all hand-written with **NumPy only**.
There are no control libraries. Matplotlib is used only for plotting.

## Approach

Each part of the assignment is treated as an experiment. I simulate the 1 kg mass-spring-damper
in Python, predict the result by hand (transfer function, damping ratio ζ, final value), and
check that the simulated graph matches. A single reusable `PID` class is extended step by step
with the real-world fixes (output limit, anti-windup, derivative on measurement, derivative filter),
so the same code demonstrates each problem and its fix.

## How to run

```bash
pip install -r requirements.txt
python run_all.py
```

The full run takes about 30 seconds. It regenerates:
- every figure in the report, saved to `figures/fig01 … fig15`
- every number quoted in the report, saved as tables in `results/results.md`

The simulation is deterministic (the sensor noise uses a fixed random seed), so anyone who
runs the code gets identical graphs and numbers.

## Files

| File | What it does |
|---|---|
| `pid.py` | Reusable `PID` class with output limits, anti-windup (clamping), derivative-on-measurement (no kick) and a low-pass filter on the D term. Each fix can be switched on or off. |
| `plants.py` | The test system `m·x'' + c·x' + k·x = F` (m = 1 kg, c = 10 N·s/m, k = 20 N/m) and the Ziegler–Nichols system `1/(s+1)³`, integrated with RK4 |
| `simulate.py` | Closed-loop simulation. The physics runs every 0.1 ms; the controller runs every `dt` and holds its output in between, as a real microcontroller does. |
| `metrics.py` | Rise time (10→90 %), overshoot, 2 % settling time and steady-state error |
| `experiments.py` | One function per figure |
| `run_all.py` | Runs everything and writes `results/results.md` |

## Figure → section map

| Figure | Report section | Shows |
|---|---|---|
| fig01_feedback_loop | 1 | The feedback loop diagram |
| fig02_open_loop_constant_force | 3 | Why holding 1 m needs a constant 20 N push |
| fig03_metrics_annotated | 4 | The four measures marked on one response |
| fig04_p_only | 5 | P only at Kp = 10, 50, 150, 300, plus the predicted offset 20/(20+Kp) |
| fig05_pi | 5 | Adding I |
| fig06_pd | 5 | Adding D, with the damping ratio ζ |
| fig07_compare_all, fig08_pid_terms | 6 | P vs PI vs PD vs PID, and each term's contribution |
| fig09_windup | 7a | 30 N limit: windup and the anti-windup fix |
| fig10_noise | 7b | ±1 mm sensor noise: the D term amplifies it, and the filter fix |
| fig11_derivative_kick | 7c | Derivative kick, and the derivative-on-measurement fix |
| fig12_update_rate | 7d | Update period 1 → 80 ms |
| fig13_zn_ultimate_gain, fig14_zn_tuned | 8 | Ziegler–Nichols on 1/(s+1)³, and an improved tuning |
| fig15_final_controller | 9 | Final controller with all fixes, under limit + noise + target changes |

## Using the controller

```python
from pid import PID
from plants import MassSpringDamper
from simulate import simulate, step_setpoint

ctrl = PID(kp=300, ki=200, kd=20, dt=0.001,
           output_limits=(-30, 30), anti_windup=True,
           derivative_on="measurement", d_filter_tau=0.01)
result = simulate(MassSpringDamper(m=1.0), ctrl, step_setpoint(1.0), t_end=3.0, noise_amp=0.001)
```

To test a "what if" case, change any argument and rerun. For example, try `m=2.0`,
`output_limits=(-20, 20)` or `noise_amp=0.005`.

## Key results (from `results/results.md`)

- **P only:** steady-state error = 20/(20+Kp). For Kp = 300 that predicts 62.5 mm, and the simulation gives 62.5 mm.
- **Final PID (Kp = 300, Ki = 200, Kd = 20):** overshoot 0.5 %, 2 % settling 242 ms, zero offset. P-only at Kp = 300 has 40.7 % overshoot and a 62.5 mm offset.
- **Windup (30 N limit):** overshoot goes from 10.8 % without anti-windup to 0 % with it, and settling time from 3.1 s to 2.0 s.
- **Noise:** filtering the D term (τ = 10 ms) cuts force jitter from 17.2 N to 1.3 N (standard deviation).
- **Kick:** the peak force on a setpoint step drops from 20,300 N to 300 N.
- **Ziegler–Nichols:** Ku = 7.99 and Pu = 3.63 s (theory: 8 and 3.628 s). Classic ZN gives 52 % overshoot; the improved tuning gives 1.1 % and settles in 3.7 s instead of 9.9 s.
