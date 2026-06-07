"""
ThermoCell - Battery Pack Thermal Model
========================================
Author : Evan McIntyre
Version: 1.0 (2026)

Models heat distribution across a cylindrical lithium-ion battery pack
using the 2D finite difference method.

Physics
-------
  1. Joule heating       Q = I^2 x R_eff           (W per cell)
  2. SoC-dependent R     R_eff = R x (1 + 0.4x(1-SoC))
  3. Fourier conduction  q = k x dT / dx            (W)
  4. Convective cooling  Q_cool = h x (T - T_amb)   (W)  [edge rows only]
  5. Temperature update  dT = (Q_net / Cm) x dt     (°C)

Usage
-----
  pip install numpy matplotlib pillow
  python thermocell.py                  # runs default (baseline) scenario
  python thermocell.py --scenario worst # runs a named preset

Outputs
-------
  thermocell_animation.gif  -- animated heat map
  thermocell_plot.png       -- final-frame summary plot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# -- Pack geometry ------------------------------------------------------------
ROWS         = 6        # cell rows
COLS         = 8        # cell columns
CELL_SPACING = 0.003    # metres between cell centres (3 mm)

# -- Cell properties (cylindrical lithium-ion, ~4680 class) ------------------
CAPACITY_AH  = 5.0      # Ah
R_INTERNAL   = 0.010    # Ohms -- internal resistance per cell
THERMAL_MASS = 50.0     # J/°C -- heat capacity per cell

# -- Thermal properties -------------------------------------------------------
K_CONDUCT    = 0.001    # W/(m·°C) -- cell-to-cell thermal conductivity

# -- Safety thresholds --------------------------------------------------------
TEMP_WARNING  = 45.0    # °C
TEMP_CRITICAL = 55.0    # °C

# -- Named presets ------------------------------------------------------------
PRESETS = {
    "baseline":   dict(c_rate=2.0, t_amb=25.0, cooling_coeff=8.0, cooling_on=True,  steps=120, dt=10.0),
    "nocooling":  dict(c_rate=2.0, t_amb=25.0, cooling_coeff=8.0, cooling_on=False, steps=120, dt=10.0),
    "hotclimate": dict(c_rate=2.0, t_amb=40.0, cooling_coeff=8.0, cooling_on=True,  steps=120, dt=10.0),
    "worst":      dict(c_rate=4.0, t_amb=40.0, cooling_coeff=0.0, cooling_on=False, steps=120, dt=10.0),
}


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def run_simulation(c_rate, t_amb, cooling_coeff, cooling_on, steps, dt):
    """Run the finite difference thermal simulation.

    Parameters
    ----------
    c_rate        : float  -- discharge C-rate (e.g. 2.0)
    t_amb         : float  -- ambient temperature (°C)
    cooling_coeff : float  -- convective cooling coefficient W/°C
    cooling_on    : bool   -- whether cooling channels are active
    steps         : int    -- number of time steps
    dt            : float  -- seconds per step

    Returns
    -------
    history : ndarray, shape (steps, ROWS, COLS)
    """
    I = c_rate * CAPACITY_AH          # current (A)
    cooling_rows = [0, ROWS - 1]

    T = np.full((ROWS, COLS), t_amb, dtype=float)
    history = np.zeros((steps, ROWS, COLS))

    for step in range(steps):
        # State-of-charge falls linearly from 1 to 0
        soc   = 1.0 - step / steps
        R_eff = R_INTERNAL * (1.0 + 0.4 * (1.0 - soc))

        dT = np.zeros((ROWS, COLS))

        # 1. Joule heating -- identical for every cell
        Q_gen  = (I ** 2) * R_eff          # W
        dT    += (Q_gen / THERMAL_MASS) * dt

        # 2. Fourier conduction to/from 4 neighbours
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        q_cond     = K_CONDUCT * (T[nr, nc] - T[r, c]) / CELL_SPACING
                        dT[r, c]  += (q_cond / THERMAL_MASS) * dt

        # 3. Convective cooling on edge rows
        #    Clamped so cells can never be pulled below ambient
        if cooling_on:
            for r in cooling_rows:
                raw_cool = (cooling_coeff * (T[r, :] - t_amb) / THERMAL_MASS) * dt
                max_cool = np.maximum(0.0, T[r, :] - t_amb)
                dT[r, :] -= np.minimum(raw_cool, max_cool)

        # 4. Update -- floor at ambient
        T = np.maximum(t_amb, T + dT)
        history[step] = T.copy()

    return history


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def make_colormap():
    """Cool blue -> yellow -> hot red."""
    colors = ["#1a6faf", "#4fb8c0", "#f5e642", "#f57c00", "#c0392b"]
    return LinearSegmentedColormap.from_list("cell_heat", colors)


def print_summary(history, params):
    final = history[-1]
    total_time = params['steps'] * params['dt']
    print("\n-- ThermoCell Results --------------------------------------------------")
    print(f"  Author           : Evan McIntyre")
    print(f"  Pack size        : {ROWS} x {COLS} cells ({ROWS * COLS} total)")
    print(f"  Discharge rate   : {params['c_rate']}C  ->  {params['c_rate'] * CAPACITY_AH:.1f} A")
    print(f"  Ambient temp     : {params['t_amb']}°C")
    print(f"  Cooling          : {'ON' if params['cooling_on'] else 'OFF'}"
          f"  (h = {params['cooling_coeff']} W/°C)")
    print(f"  Duration         : {total_time:.0f} s  ({total_time/60:.1f} min)")
    print(f"  Final max temp   : {final.max():.1f}°C", end="")
    if   final.max() > TEMP_CRITICAL: print("  [!!] CRITICAL -- thermal runaway risk")
    elif final.max() > TEMP_WARNING:  print("  [!]  WARNING")
    else:                              print("  [ok] Safe")
    print(f"  Final avg temp   : {final.mean():.1f}°C")
    print(f"  Temp spread      : {final.max() - final.min():.1f}°C")
    print("------------------------------------------------------------------------\n")


def animate_and_save(history, params):
    """Render animated heat map + temperature chart, save GIF and PNG."""
    cmap  = make_colormap()
    vmin  = params['t_amb'] - 2
    vmax  = max(float(history.max()), TEMP_CRITICAL + 5)
    DT    = params['dt']
    steps = params['steps']
    times = np.arange(steps) * DT

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")

    # -- Left: heat map -------------------------------------------------------
    ax1 = axes[0]
    im  = ax1.imshow(history[0], cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest", aspect="equal")

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax1.set_xlabel("Column", color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("Row",    color="#aaaaaa", fontsize=9)
    ax1.tick_params(colors="#aaaaaa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    title = ax1.set_title("", color="white", fontsize=11, pad=10)

    # Cooling channel markers
    if params['cooling_on']:
        for r in [0, ROWS - 1]:
            ax1.annotate("[COOL]", xy=(-1.2, r), xycoords="data",
                         color="#4fb8c0", fontsize=8, va="center")

    # Author watermark
    fig.text(0.99, 0.01, "Evan McIntyre -- ThermoCell v1.0",
             ha="right", va="bottom", fontsize=7, color="#444444",
             fontstyle="italic")

    # -- Right: temperature vs time -------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f0f")

    max_t = history.max(axis=(1, 2))
    avg_t = history.mean(axis=(1, 2))
    min_t = history.min(axis=(1, 2))

    ax2.plot(times, max_t, color="#c0392b", lw=1.8, label="Max")
    ax2.plot(times, avg_t, color="#4fb8c0", lw=1.8, label="Avg")
    ax2.plot(times, min_t, color="#1a6faf", lw=1.8, label="Min")
    ax2.axhline(TEMP_WARNING,  color="#f5e642", lw=1, ls="--",
                label=f"Warning ({TEMP_WARNING}°C)")
    ax2.axhline(TEMP_CRITICAL, color="#e74c3c", lw=1, ls="--",
                label=f"Critical ({TEMP_CRITICAL}°C)")

    ax2.set_xlabel("Time (s)",         color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Temperature (°C)", color="#aaaaaa", fontsize=9)
    ax2.set_title("Temperature over discharge", color="white", fontsize=11, pad=10)
    ax2.tick_params(colors="#aaaaaa")
    ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444",
               labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    vline = ax2.axvline(x=0, color="white", lw=0.8, alpha=0.5)

    # -- Animation ------------------------------------------------------------
    def update(frame):
        im.set_data(history[frame])
        t   = frame * DT
        soc = max(0, 100 - round(100 * frame / steps))
        status = ""
        mx = history[frame].max()
        if   mx > TEMP_CRITICAL: status = "  [!!] CRITICAL"
        elif mx > TEMP_WARNING:  status = "  [!] WARNING"
        title.set_text(
            f"t = {int(t)} s  |  SoC = {soc}%  |  Max: {mx:.1f}°C{status}"
        )
        vline.set_xdata([t, t])
        return im, title, vline

    ani = animation.FuncAnimation(fig, update, frames=steps,
                                  interval=80, blit=True)
    plt.tight_layout()

    gif_path = "thermocell_animation.gif"
    print(f"Saving animation -> {gif_path}  (may take ~30 s) ...")
    ani.save(gif_path, writer="pillow", fps=15, dpi=100)
    print("Done.")

    png_path = "thermocell_plot.png"
    update(steps - 1)
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Final frame -> {png_path}")

    plt.show()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ThermoCell - Battery Pack Thermal Model by Evan McIntyre"
    )
    parser.add_argument(
        "--scenario",
        choices=list(PRESETS.keys()),
        default="baseline",
        help="Named scenario to run (default: baseline)"
    )
    parser.add_argument("--crate",   type=float, help="Override C-rate")
    parser.add_argument("--tamb",    type=float, help="Override ambient temp (°C)")
    parser.add_argument("--cooling", type=int,   choices=[0, 1],
                        help="Override cooling: 1=on, 0=off")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    params = dict(PRESETS[args.scenario])

    if args.crate   is not None: params["c_rate"]    = args.crate
    if args.tamb    is not None: params["t_amb"]      = args.tamb
    if args.cooling is not None: params["cooling_on"] = bool(args.cooling)

    print(f"\nThermoCell v1.0  --  Evan McIntyre")
    print(f"Running scenario: {args.scenario.upper()}")
    print(f"  C-rate: {params['c_rate']}C  |  Ambient: {params['t_amb']}°C  |  "
          f"Cooling: {'ON' if params['cooling_on'] else 'OFF'}\n")

    history = run_simulation(**params)
    print_summary(history, params)
    animate_and_save(history, params)
