"""
Every experiment / figure in the report. Run `python run_all.py` to regenerate everything.

Each function saves one figure to figures/ and returns rows for the results tables.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pid import PID
from plants import MassSpringDamper, ThirdOrderLag
from simulate import simulate, step_setpoint
from metrics import step_metrics

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

DT = 0.001                      # controller update period (1 kHz) unless stated otherwise
FINAL = dict(kp=300, ki=200, kd=20)   # final gains chosen in Section 6
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "axes.titleweight": "bold",
                     "legend.fontsize": 9})


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path)


def run_block(kp, ki=0.0, kd=0.0, t_end=3.0, dt=DT, sp=step_setpoint(1.0), noise=0.0,
              plant=None, **pid_kw):
    plant = plant or MassSpringDamper()
    return simulate(plant, PID(kp, ki, kd, dt=dt, **pid_kw), sp, t_end, noise_amp=noise)


def setpoint_line(ax, value=1.0, label="setpoint (1 m)"):
    ax.axhline(value, color="k", ls="--", lw=1, label=label)


# ----------------------------------------------------------------------------------------
# Section 1: feedback loop diagram
# ----------------------------------------------------------------------------------------
def fig_feedback_loop():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="k", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 0.18,
                    label, ha="center", fontsize=9, color="#333")

    ax.text(0.35, 2.3, "Setpoint\n(target, 1 m)", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Circle((1.5, 2.3), 0.22, fc="white", ec="k", lw=1.2))
    ax.text(1.5, 2.3, "Σ", ha="center", va="center", fontsize=12)
    ax.text(1.28, 2.62, "+", fontsize=10); ax.text(1.62, 1.85, "−", fontsize=12)
    arrow(0.9, 2.3, 1.28, 2.3)
    box(2.4, 1.9, 2.0, 0.8, "Controller\n(PID)", "#cfe2ff")
    arrow(1.72, 2.3, 2.4, 2.3, "error e", ly=2.5)
    box(5.4, 1.9, 2.0, 0.8, "Plant\n(mass-spring-damper)", "#ffe5b4")
    arrow(4.45, 2.3, 5.4, 2.3, "force u (N)", ly=2.5)
    arrow(7.45, 2.3, 9.6, 2.3, "position x (m)", lx=8.6, ly=2.5)
    box(5.4, 0.4, 2.0, 0.7, "Sensor", "#d4edda")
    ax.plot([8.6, 8.6], [2.3, 0.75], "k", lw=1.5)
    arrow(8.6, 0.75, 7.45, 0.75)
    ax.plot([5.4, 1.5], [0.75, 0.75], "k", lw=1.5)
    arrow(1.5, 0.75, 1.5, 2.08, "measured x", lx=0.85, ly=1.3)
    ax.set_title("Fig 1 - The feedback loop: measure, compare with the target, correct, repeat")
    save(fig, "fig01_feedback_loop.png")


# ----------------------------------------------------------------------------------------
# Section 3: why the spring needs a constant push
# ----------------------------------------------------------------------------------------
def fig_open_loop():
    t_end, h = 3.0, 1e-4
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for F, c in zip([0, 10, 20, 30], COLORS):
        plant = MassSpringDamper()
        from plants import rk4_step
        ts, xs = [], []
        for i in range(int(t_end / h)):
            ts.append(i * h); xs.append(plant.output()); rk4_step(plant, F, h)
        ax.plot(ts, xs, color=c, label=f"constant force {F} N  →  x = {F/20:.1f} m")
    setpoint_line(ax)
    ax.set(title="Fig 2 - No controller: a constant force F holds the block at x = F / k",
           xlabel="time (s)", ylabel="position x (m)")
    ax.legend(loc="lower right")
    save(fig, "fig02_open_loop_constant_force.png")


# ----------------------------------------------------------------------------------------
# Section 4: the four measures, marked on one graph
# ----------------------------------------------------------------------------------------
def fig_metrics_annotated():
    r = run_block(150, t_end=1.5)
    m = step_metrics(r["t"], r["y"])
    t, y, yf = r["t"], r["y"], m["final_value"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, y, color=COLORS[0], lw=2, label="position (P-only, Kp = 150)")
    setpoint_line(ax)
    ax.axhspan(0.98 * yf, 1.02 * yf, color="green", alpha=0.12, label="±2 % band around final value")
    ax.axhline(yf, color="gray", ls=":", lw=1)

    t10 = t[np.where(y >= 0.1 * yf)[0][0]]; t90 = t[np.where(y >= 0.9 * yf)[0][0]]
    ax.plot([t10, t90], [0.1 * yf, 0.9 * yf], "o", color=COLORS[1])
    ax.annotate("", xy=(t90, 0.15), xytext=(t10, 0.15), arrowprops=dict(arrowstyle="<->", color=COLORS[1]))
    ax.text(t90 + 0.03, 0.12, f"① rise time\n{m['rise_time']*1000:.0f} ms",
            ha="left", color=COLORS[1], fontsize=9)

    ax.annotate("", xy=(m["t_peak"], m["peak"]), xytext=(m["t_peak"], yf),
                arrowprops=dict(arrowstyle="<->", color=COLORS[3]))
    ax.text(m["t_peak"] + 0.04, m["peak"] - 0.02, f"② overshoot\n{m['overshoot']:.1f} %",
            color=COLORS[3], fontsize=9, va="center")

    ax.axvline(m["settling_time"], color=COLORS[2], ls="--", lw=1)
    ax.text(m["settling_time"] + 0.02, 0.45, f"③ settling time (2 %)\n{m['settling_time']*1000:.0f} ms",
            color=COLORS[2], fontsize=9)

    ax.annotate("", xy=(1.35, 1.0), xytext=(1.35, yf), arrowprops=dict(arrowstyle="<->", color=COLORS[4]))
    ax.text(1.33, 0.72, f"④ steady-state\nerror {m['ss_error']*1000:.0f} mm",
            color=COLORS[4], fontsize=9, va="center", ha="right")
    ax.set(title="Fig 3 - The four measures of a step response", xlabel="time (s)",
           ylabel="position x (m)", ylim=(0, 1.25))
    ax.legend(loc="lower right")
    save(fig, "fig03_metrics_annotated.png")
    return m


# ----------------------------------------------------------------------------------------
# Section 5: each part alone
# ----------------------------------------------------------------------------------------
def second_order_prediction(kp, kd=0.0, m=1.0, c=10.0, k=20.0):
    """Closed loop (P or PD, D on error):  m x'' + (c+Kd) x' + (k+Kp) x = Kd r' + Kp r"""
    wn = np.sqrt((k + kp) / m)
    zeta = (c + kd) / (2 * np.sqrt(m * (k + kp)))
    os_pct = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) if zeta < 1 else 0.0
    return dict(final=kp / (k + kp), ess=k / (k + kp), wn=wn, zeta=zeta, os_poles=os_pct)


def fig_p_only():
    gains = [10, 50, 150, 300]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kp, c in zip(gains, COLORS):
        r = run_block(kp, t_end=2.0)
        m = step_metrics(r["t"], r["y"]); th = second_order_prediction(kp)
        ax.plot(r["t"], r["y"], color=c, label=f"Kp = {kp}")
        ax.axhline(th["final"], color=c, ls=":", lw=1)
        ax.text(1.98, th["final"] + 0.015, f"predicted {th['final']:.3f} m", color=c,
                ha="right", fontsize=8)
        rows.append(dict(Kp=kp, ess_pred=th["ess"], ess_sim=m["ss_error"],
                         zeta=th["zeta"], os_pred=th["os_poles"], os_sim=m["overshoot"],
                         rise=m["rise_time"], settle=m["settling_time"]))
    setpoint_line(ax)
    ax.set(title="Fig 4 - P-only: higher Kp gets closer to 1 m but never reaches it, and bounces more",
           xlabel="time (s)", ylabel="position x (m)", ylim=(0, 1.5))
    ax.legend(loc="upper right", ncol=5)
    save(fig, "fig04_p_only.png")
    return rows


def fig_pi():
    kis = [0, 50, 200, 1000]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ki, c in zip(kis, COLORS):
        r = run_block(300, ki, t_end=6.0)
        m = step_metrics(r["t"], r["y"])
        ax.plot(r["t"], r["y"], color=c, label=f"Kp = 300, Ki = {ki}")
        rows.append(dict(Ki=ki, **m))
    setpoint_line(ax)
    ax.set(title="Fig 5 - Adding I: the offset disappears; too much Ki adds overshoot",
           xlabel="time (s)", ylabel="position x (m)", ylim=(0, 1.7))
    ax.legend(loc="upper right", ncol=2)
    save(fig, "fig05_pi.png")
    return rows


def fig_pd():
    kds = [0, 10, 20, 30]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kd, c in zip(kds, COLORS):
        r = run_block(300, 0, kd, t_end=1.5)
        m = step_metrics(r["t"], r["y"]); th = second_order_prediction(300, kd)
        ax.plot(r["t"], r["y"], color=c, label=f"Kp = 300, Kd = {kd}  (ζ = {th['zeta']:.2f})")
        rows.append(dict(Kd=kd, zeta=th["zeta"], os_pred=th["os_poles"], **m))
    setpoint_line(ax)
    ax.axhline(300 / 320, color="gray", ls=":", lw=1)
    ax.text(1.48, 300 / 320 - 0.06, "P-only final value 0.9375 m", ha="right", fontsize=8, color="gray")
    ax.set(title="Fig 6 - Adding D: damping ratio ζ rises, so bouncing dies out",
           xlabel="time (s)", ylabel="position x (m)", ylim=(0, 1.5))
    ax.legend(loc="upper right")
    save(fig, "fig06_pd.png")
    return rows


# ----------------------------------------------------------------------------------------
# Section 6: all together
# ----------------------------------------------------------------------------------------
def fig_compare_all():
    configs = [("P", 300, 0, 0), ("PI", 300, 200, 0), ("PD", 300, 0, 20), ("PID", 300, 200, 20)]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, kp, ki, kd), c in zip(configs, COLORS):
        r = run_block(kp, ki, kd, t_end=5.0)
        m = step_metrics(r["t"], r["y"])
        ax.plot(r["t"], r["y"], color=c, lw=2 if name == "PID" else 1.3,
                label=f"{name}  (Kp={kp}, Ki={ki}, Kd={kd})")
        rows.append(dict(name=name, Kp=kp, Ki=ki, Kd=kd, peak_u=r["u"].max(), **m))
    setpoint_line(ax)
    ax.set(title="Fig 7 - P vs PI vs PD vs PID on the same block", xlabel="time (s)",
           ylabel="position x (m)", xlim=(0, 3.0), ylim=(0, 1.5))
    ax.legend(loc="upper right")
    save(fig, "fig07_compare_all.png")

    # what each term contributes in the final PID
    r = run_block(**FINAL, t_end=1.0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    a1.plot(r["t"], r["y"], color=COLORS[0], label="position"); setpoint_line(a1)
    a1.set(ylabel="position x (m)", title="Fig 8 - Final PID: what P, I and D each contribute")
    a1.legend(loc="lower right")
    a2.plot(r["t"], r["p"], label="P term"); a2.plot(r["t"], r["i"], label="I term")
    a2.plot(r["t"], r["d"], label="D term"); a2.plot(r["t"], r["u"], "k", lw=1, label="total force u")
    a2.axhline(20, color="gray", ls=":", lw=1); a2.text(0.98, 23, "20 N needed to hold 1 m", ha="right", fontsize=8)
    a2.set(xlabel="time (s)", ylabel="force (N)"); a2.legend(loc="upper right", ncol=4)
    save(fig, "fig08_pid_terms.png")
    return rows


# ----------------------------------------------------------------------------------------
# Section 7: real-world problems and fixes
# ----------------------------------------------------------------------------------------
def fig_windup():
    lim = (-30, 30)
    runs = {
        "no limit (ideal)": run_block(**FINAL, t_end=6.0),
        "30 N limit, no anti-windup": run_block(**FINAL, t_end=6.0, output_limits=lim),
        "30 N limit + anti-windup (clamping)": run_block(**FINAL, t_end=6.0, output_limits=lim, anti_windup=True),
    }
    rows = []
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for (name, r), c in zip(runs.items(), [COLORS[2], COLORS[3], COLORS[0]]):
        m = step_metrics(r["t"], r["y"])
        rows.append(dict(name=name, **m))
        a1.plot(r["t"], r["y"], color=c, label=f"{name}: overshoot {m['overshoot']:.1f} %")
        a2.plot(r["t"], r["u"], color=c, label=name)
        a3.plot(r["t"], r["i"], color=c, label=name)
    setpoint_line(a1)
    a1.set(ylabel="position x (m)", title="Fig 9 - Integral windup with a 30 N motor, and the fix")
    a1.legend(loc="lower right", fontsize=8)
    a2.set(ylabel="force u (N)", ylim=(-40, 60)); a2.axhline(30, color="k", ls=":", lw=1)
    a2.legend(loc="upper right", fontsize=8)
    a3.set(ylabel="I term (N)", xlabel="time (s)")
    a3.legend(loc="upper right", fontsize=8)
    save(fig, "fig09_windup.png")
    return rows


def fig_noise():
    noise = 0.001
    runs = {
        "D on raw noisy signal": run_block(**FINAL, noise=noise, derivative_on="measurement"),
        "D low-pass filtered (τ = 10 ms)": run_block(**FINAL, noise=noise, derivative_on="measurement",
                                                    d_filter_tau=0.01),
    }
    rows = []
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for (name, r), c in zip(runs.items(), [COLORS[3], COLORS[0]]):
        ss = r["t"] > 1.0
        rows.append(dict(name=name, u_std=r["u"][ss].std(), u_p2p=np.ptp(r["u"][ss]),
                         x_std_mm=1000 * r["y"][ss].std(), **step_metrics(r["t"], r["y"])))
        a2.plot(r["t"], r["u"], color=c, lw=0.7,
                label=f"{name}: force std {r['u'][ss].std():.1f} N")
        a1.plot(r["t"], r["y"], color=c, lw=1, label=name)
    setpoint_line(a1)
    a1.set(ylabel="position x (m)",
           title="Fig 10 - ±1 mm sensor noise: the D term amplifies it; a filter fixes it")
    a1.legend(loc="lower right")
    a2.set(xlabel="time (s)", ylabel="force u (N)", ylim=(-60, 100)); a2.legend(loc="upper right")
    save(fig, "fig10_noise.png")
    return rows


def fig_kick():
    sp = step_setpoint(1.0, t_step=0.5)
    runs = {"D on error (textbook)": run_block(**FINAL, t_end=2.5, sp=sp, derivative_on="error"),
            "D on measurement (fix)": run_block(**FINAL, t_end=2.5, sp=sp, derivative_on="measurement")}
    rows = []
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for (name, r), c, ls in zip(runs.items(), [COLORS[3], COLORS[0]], ["-", "--"]):
        rows.append(dict(name=name, peak_u=r["u"].max(), **step_metrics(r["t"], r["y"], t_step=0.5)))
        a1.plot(r["t"], r["y"], color=c, ls=ls, label=name)
        a2.plot(r["t"], r["u"], color=c, ls=ls, label=f"{name}: peak {r['u'].max():,.0f} N")
    a1.plot(r["t"], r["sp"], "k:", lw=1, label="setpoint")
    a1.set(ylabel="position x (m)", title="Fig 11 - Derivative kick when the target jumps 0 → 1 m at t = 0.5 s")
    a1.legend(loc="lower right")
    a2.set_yscale("symlog", linthresh=100)
    a2.set(xlabel="time (s)", ylabel="force u (N, log scale)"); a2.legend(loc="upper right")
    save(fig, "fig11_derivative_kick.png")
    return rows


def fig_sample_time():
    dts = [0.001, 0.02, 0.05, 0.065, 0.08]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for dt, c in zip(dts, COLORS):
        r = run_block(**FINAL, dt=dt, t_end=3.0, derivative_on="measurement")
        m = step_metrics(r["t"], r["y"])
        rows.append(dict(dt_ms=dt * 1000, **m))
        ax.step(r["t"], r["y"], where="post", color=c,
                label=f"update every {dt*1000:g} ms: " + (f"overshoot {m['overshoot']:.0f} %"
                      if not np.isnan(m['settling_time']) else "never settles"))
    setpoint_line(ax)
    ax.set(title="Fig 12 - Slower controller updates: fine, then oscillating, then unstable",
           xlabel="time (s)", ylabel="position x (m)", ylim=(-0.5, 2.5))
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig12_update_rate.png")
    return rows


# ----------------------------------------------------------------------------------------
# Section 8: Ziegler-Nichols on 1/(s+1)^3
# ----------------------------------------------------------------------------------------
def _peaks(t, y):
    i = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return t[i], y[i]


def find_ultimate_gain(lo=1.0, hi=20.0, iters=22):
    """Raise Kp (bisection) until the P-only loop just keeps swinging: peak amplitude neither grows nor decays."""
    def growth(kp):
        r = simulate(ThirdOrderLag(), PID(kp, dt=0.001), step_setpoint(1.0), 60.0, dt_sim=0.001)
        tp, yp = _peaks(r["t"], r["y"] - np.mean(r["y"][len(r["y"]) // 2:]))
        return yp[-2] / yp[1] if len(yp) > 3 else 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if growth(mid) > 1.0:
            hi = mid
        else:
            lo = mid
    ku = 0.5 * (lo + hi)
    r = simulate(ThirdOrderLag(), PID(ku, dt=0.001), step_setpoint(1.0), 40.0, dt_sim=0.001)
    tp, _ = _peaks(r["t"], r["y"])
    pu = np.mean(np.diff(tp[2:]))
    return ku, pu


def fig_ziegler_nichols():
    ku, pu = find_ultimate_gain()
    ku_th, pu_th = 8.0, 2 * np.pi / np.sqrt(3)

    # Plot the search
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for kp, c in zip([4, ku, 9], [COLORS[0], COLORS[2], COLORS[3]]):
        r = simulate(ThirdOrderLag(), PID(kp, dt=0.001), step_setpoint(1.0), 25.0, dt_sim=0.001)
        lbl = {4: "Kp = 4: swings die out", 9: "Kp = 9: swings grow (unstable)"}.get(kp, f"Kp = Ku = {ku:.2f}: steady swing, period Pu = {pu:.2f} s")
        ax.plot(r["t"], r["y"], color=c, label=lbl)
    ax.set(title="Fig 13 - Ziegler-Nichols step 1: raise Kp until the output swings steadily",
           xlabel="time (s)", ylabel="output y (-)", ylim=(-1.0, 3.0))
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, "fig13_zn_ultimate_gain.png")

    rules = {
        "Classic ZN PID":  (0.6 * ku, pu / 2, pu / 8),
        "ZN 'no overshoot' rule": (0.2 * ku, pu / 2, pu / 3),
    }
    configs = {}
    for name, (kp, ti, td) in rules.items():
        configs[name] = (kp, kp / ti, kp * td)
    # Hand refinement starting from classic ZN (see report Section 8):
    # ZN aims for "quarter-amplitude decay", which is aggressive. Keep ZN's Kd (damping),
    # cut Ki to ~1/4 (the integral is the main source of overshoot), and lower Kp to ~0.3 Ku.
    zn_kd = configs["Classic ZN PID"][2]
    configs["Improved (hand-tuned from ZN)"] = (2.5, 0.6, round(zn_kd, 2))

    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, (kp, ki, kd)), c in zip(configs.items(), [COLORS[3], COLORS[1], COLORS[0]]):
        r = simulate(ThirdOrderLag(), PID(kp, ki, kd, dt=0.001, derivative_on="measurement"),
                     step_setpoint(1.0), 25.0, dt_sim=0.001)
        m = step_metrics(r["t"], r["y"])
        rows.append(dict(name=name, Kp=kp, Ki=ki, Kd=kd, **m))
        ax.plot(r["t"], r["y"], color=c,
                label=f"{name}: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} → overshoot {m['overshoot']:.0f} %")
    setpoint_line(ax, label="setpoint (1)")
    ax.set(title="Fig 14 - Ziegler-Nichols gains on 1/(s+1)³, and an improved tuning",
           xlabel="time (s)", ylabel="output y (-)", ylim=(0, 1.7))
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig14_zn_tuned.png")
    return dict(ku=ku, pu=pu, ku_theory=ku_th, pu_theory=pu_th), rows


# ----------------------------------------------------------------------------------------
# Section 9: the reusable controller with every fix, on the test system
# ----------------------------------------------------------------------------------------
def fig_final_controller():
    def sp(t):
        return 0.0 if t < 0.2 else (1.0 if t < 4.0 else 0.5)
    ctrl = PID(**FINAL, dt=DT, output_limits=(-30, 30), anti_windup=True,
               derivative_on="measurement", d_filter_tau=0.01)
    r = simulate(MassSpringDamper(), ctrl, sp, 6.0, noise_amp=0.001)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(r["t"], r["sp"], "k--", lw=1, label="setpoint")
    a1.plot(r["t"], r["y_meas"], color="#bbb", lw=0.6, label="noisy sensor reading")
    a1.plot(r["t"], r["y"], color=COLORS[0], lw=1.6, label="true position")
    a1.set(ylabel="position x (m)",
           title="Fig 15 - Final PID (all fixes on): 30 N limit, ±1 mm noise, target 0 → 1 → 0.5 m")
    a1.legend(loc="lower right")
    a2.plot(r["t"], r["u"], color=COLORS[3], lw=0.8, label="motor force u")
    a2.axhline(30, color="k", ls=":", lw=1); a2.axhline(-30, color="k", ls=":", lw=1, label="±30 N limit")
    a2.set(xlabel="time (s)", ylabel="force u (N)", ylim=(-40, 40)); a2.legend(loc="lower right")
    save(fig, "fig15_final_controller.png")
    m1 = step_metrics(r["t"][r["t"] < 4.0], r["y"][r["t"] < 4.0], t_step=0.2)
    return [dict(name="final controller, 0 → 1 m step", **m1)]
