"""
Double Pendulum Analysis Script
================================
Analyses Tracker-exported CSV data for N trials of a double pendulum.

Expected CSV format — Tracker's "#multi" export with omega (one file per trial):

    #multi:
    ,m2,,,m1,,,        ← object order may vary; loader detects automatically
    t,x,y,ω,x,y,ω,    ← ω column added for each mass
    0.000,...
    ...

The loader reads the object-name row to disambiguate columns and produces:
    t, x1, y1, omega1, x2, y2, omega2
where omega is Tracker's computed angular velocity in deg/s.
If omega columns are absent the script falls back to numerical differentiation.
Note: Tracker may leave the first omega value empty — handled automatically.

Units: time in seconds, positions in metres (as Tracker exports them).
The pivot of arm 1 is at the origin (0, 0), as set in Tracker.

What this script produces
--------------------------
1. Angles θ₁(t) and θ₂(t) for every trial.
2. Angular velocities ω₁(t) and ω₂(t) via numerical differentiation.
3. Angle vs. time plots (all trials overlaid, both arms).
4. Phase-space plots  ω vs. θ  for both arms.
5. Trajectory-divergence plot  Δθ(t)  between every pair of trials.
6. Estimate of the largest Lyapunov exponent λ (linear-fit on log divergence).
7. A summary figure combining the key results.

Usage
------
    python double_pendulum_analysis.py

Edit the CONFIG section below to match your file paths and setup.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from scipy.signal import savgol_filter
from itertools import combinations

# ─────────────────────────── CONFIG ───────────────────────────────────────────

# Glob pattern that matches all trial CSV files in order.
# Put your CSV files in the same folder as this script, or give the full path.
CSV_PATTERN = "trial_*.csv"

# Names Tracker uses for your two tracked masses (row 2 of the CSV).
# These are matched case-insensitively and stripped of whitespace.
MASS1_NAME = "m1"   # first  tracked object → upper arm endpoint
MASS2_NAME = "m2"   # second tracked object → lower arm endpoint

# Savitzky-Golay smoothing for angle and velocity (set SMOOTH=False to skip)
SMOOTH        = True
SG_WINDOW     = 11   # must be odd; increase for noisier data
SG_POLY       = 3    # polynomial order

# Lyapunov: time window (seconds) over which to fit the exponential growth
LYAP_T_MIN = 0.5    # start of fitting window
LYAP_T_MAX = 1.5    # end   of fitting window  (adjust to your data length)

# Output folder for figures
OUTPUT_DIR = "figures"

# ──────────────────────────────────────────────────────────────────────────────


def load_trial(path):
    """Load one Tracker "#multi" CSV file.

    Tracker's multi-object export has this structure:
        Line 1:  #multi:                      ← comment, skipped
        Line 2:  ,m1,,m2,                     ← object-name row
        Line 3:  t,x,y,x,y                    ← column-name row
        Line 4+: data rows

    Because both masses share the same column names ("x", "y") we read the
    object-name row manually and build unique keys:
        t, x1, y1, x2, y2
    so the rest of the pipeline can use them unambiguously.
    """
    with open(path, encoding="utf-8-sig") as fh:
        raw_lines = [l.rstrip("\n") for l in fh if l.strip()]

    # ── detect delimiter ──────────────────────────────────────────────────────
    # Use the column-header row (first non-comment line that contains "t")
    delim = "\t"
    for l in raw_lines:
        if l.startswith("#"):
            continue
        delim = "\t" if "\t" in l else ","
        break

    def split(line):
        return [c.strip() for c in line.split(delim)]

    # ── separate comment / object-name / column-name / data lines ────────────
    comment_lines = [l for l in raw_lines if l.startswith("#")]
    data_lines    = [l for l in raw_lines if not l.startswith("#")]

    # data_lines[0] = object-name row  e.g.  ",m1,,m2,"
    # data_lines[1] = column-name row  e.g.  "t,x,y,x,y"
    # data_lines[2+]= numeric data
    if len(data_lines) < 3:
        raise ValueError(f"File too short or unexpected format: {path}")

    obj_row = split(data_lines[0])   # ['', 'm1', '', 'm2', '']
    col_row = split(data_lines[1])   # ['t', 'x', 'y', 'x', 'y']

    # ── build unique column names using the object-name row ──────────────────
    # Walk the object-name row: whenever a non-empty name appears, remember it
    # as the "current object" for subsequent columns until the next name change.
    unique_cols = []
    current_obj = ""
    for obj_cell, col_cell in zip(obj_row, col_row):
        obj_cell = obj_cell.strip()
        col_cell = col_cell.strip()
        if obj_cell:                     # new object name encountered
            current_obj = obj_cell.lower()

        if col_cell.lower() == "t":
            unique_cols.append("t")
        elif current_obj in (MASS1_NAME.lower(),):
            unique_cols.append(f"{col_cell}1")   # x1, y1, …
        elif current_obj in (MASS2_NAME.lower(),):
            unique_cols.append(f"{col_cell}2")   # x2, y2, …
        else:
            unique_cols.append(col_cell)          # fallback

    # ── parse numeric data ────────────────────────────────────────────────────
    arrays = {c: [] for c in unique_cols}
    for line in data_lines[2:]:
        vals = split(line)
        for col, val in zip(unique_cols, vals):
            try:
                arrays[col].append(float(val))
            except ValueError:
                arrays[col].append(np.nan)

    result = {k: np.array(v) for k, v in arrays.items()}

    # ── sanity check ──────────────────────────────────────────────────────────
    for required in ("t", "x1", "y1", "x2", "y2"):
        if required not in result:
            raise KeyError(
                f"Column '{required}' not found after parsing {path}.\n"
                f"  Object-name row : {obj_row}\n"
                f"  Column-name row : {col_row}\n"
                f"  Parsed columns  : {list(result.keys())}\n"
                f"  → Check that MASS1_NAME='{MASS1_NAME}' and "
                f"MASS2_NAME='{MASS2_NAME}' match your CSV exactly."
            )

    # Remove ghost columns from trailing commas in Tracker export
    # (e.g. a column named "" or "1" or "2" with all-NaN values)
    for ghost in [k for k in list(result.keys())
                  if k.strip() in ("", "1", "2") or
                     all(np.isnan(v) for v in result[k])]:
        result.pop(ghost, None)

    # Detect omega columns — Tracker names them "ω" which after our suffix
    # renaming becomes "ω1" and "ω2". Also handle ascii variants.
    def _is_omega(key, suffix):
        k = key.lower().replace("|","").replace(" ","")
        return k in (f"\u03c9{suffix}", f"omega{suffix}", f"w{suffix}",
                     f"|\u03c9|{suffix}")
    for suffix in ("1", "2"):
        for raw_key in list(result.keys()):
            if _is_omega(raw_key, suffix):
                arr = result.pop(raw_key).copy()
                # Replace NaN / empty first frame with 0
                arr = np.where(np.isnan(arr), 0.0, arr)
                result[f"omega_tracker{suffix}"] = arr
                break

    return result


def smooth(arr, window=SG_WINDOW, poly=SG_POLY):
    """Apply Savitzky-Golay filter; fall back gracefully if array is too short."""
    if not SMOOTH or len(arr) < window:
        return arr
    return savgol_filter(arr, window, poly)


def differentiate(t, arr):
    """Numerical derivative via central differences, then smooth."""
    dadt = np.gradient(arr, t)
    return smooth(dadt)


def compute_theta1(data):
    """Compute theta1 from positions (arm 1 never spins — no aliasing risk).

    theta1 = atan2(x1, -y1) measured from the downward vertical.
    Positions are smoothed before atan2, then unwrapped.
    """
    x1, y1 = data["x1"], data["y1"]
    if SMOOTH and len(x1) >= SG_WINDOW:
        x1 = savgol_filter(x1, SG_WINDOW, SG_POLY)
        y1 = savgol_filter(y1, SG_WINDOW, SG_POLY)
    theta1 = np.arctan2(x1, -y1)
    theta1 = np.unwrap(theta1)
    return smooth(theta1)


def compute_theta2_and_omega2(data):
    """Compute theta2 and omega2 for the lower arm.

    WHAT TRACKER'S OMEGA COLUMN ACTUALLY IS:
        Tracker computes omega as a finite difference of the raw atan2 angle.
        Because atan2 wraps at ±pi, Tracker's omega has large spikes each time
        arm 2 crosses the ±180 degree boundary. These spikes cancel when
        integrated, so the integral does NOT recover the true cumulative angle.
        Tracker's omega is therefore NOT suitable for integration.

    WHAT TRACKER'S OMEGA IS GOOD FOR:
        Despite the integration problem, Tracker's omega gives the correct
        INSTANTANEOUS angular velocity at each frame (away from the ±pi
        boundary). It is therefore used directly as omega2 for the phase-space
        plot y-axis, giving a more accurate velocity than numerical
        differentiation of our noisy theta2.

    THETA2 METHOD:
        Raw atan2 on positions → np.unwrap → smooth.
        This is the best available method at 30fps. The cumulative rotation
        count may be slightly off for the fastest spinning frames but the
        qualitative behaviour is correct.
    """
    t  = data["t"]
    has_omega = "omega_tracker2" in data

    # theta2: always from positions (Tracker omega cannot be integrated)
    x2, y2 = data["x2"], data["y2"]
    x1, y1 = data["x1"], data["y1"]
    theta2_raw = np.arctan2(x2 - x1, -(y2 - y1))
    theta2 = smooth(np.unwrap(theta2_raw))

    # omega2: use Tracker's direct measurement if available (more accurate
    # than differentiating our noisy theta2), else numerical diff
    if has_omega:
        omega2 = smooth(data["omega_tracker2"] * (np.pi / 180.0))
    else:
        omega2 = differentiate(t, theta2)

    return theta2, omega2


def process_trial(path):
    """Full pipeline for one trial file."""
    data   = load_trial(path)
    t      = data["t"]

    # theta1: from smoothed positions, unwrapped (arm 1 never spins)
    theta1 = compute_theta1(data)

    # omega1: use Tracker's direct measurement if available, else diff
    if "omega_tracker1" in data:
        omega1 = smooth(data["omega_tracker1"] * (np.pi / 180.0))
    else:
        omega1 = differentiate(t, theta1)

    # theta2 and omega2
    theta2, omega2 = compute_theta2_and_omega2(data)

    has_omega = "omega_tracker2" in data
    return dict(t=t, theta1=theta1, theta2=theta2,
                omega1=omega1, omega2=omega2, path=path,
                has_omega=has_omega)


def lyapunov_estimate(trials, t_min=LYAP_T_MIN, t_max=LYAP_T_MAX):
    """Estimate the largest Lyapunov exponent from pairwise divergence.

    Method
    -------
    1. For every pair of trials interpolate θ₁ onto a common time grid.
    2. Compute |Δθ₁(t)| = |θ₁_i(t) − θ₁_j(t)|.
    3. Average log(|Δθ|) over all pairs.
    4. Fit a line to ⟨log|Δθ|⟩ vs t in the window [t_min, t_max].
       The slope is λ (Lyapunov exponent, units: 1/s).
    """
    # Common time grid
    t_common = trials[0]["t"]
    for tr in trials[1:]:
        t_start = max(t_common[0],  tr["t"][0])
        t_end   = min(t_common[-1], tr["t"][-1])
        t_common = np.linspace(t_start, t_end, min(len(t_common), len(tr["t"])))

    # Interpolate θ₁ for every trial onto common grid
    theta1_interp = []
    for tr in trials:
        th = np.interp(t_common, tr["t"], tr["theta1"])
        theta1_interp.append(th)

    pairs    = list(combinations(range(len(trials)), 2))
    log_divs = []

    for i, j in pairs:
        delta = np.abs(theta1_interp[i] - theta1_interp[j])
        delta = np.where(delta < 1e-10, 1e-10, delta)   # avoid log(0)
        log_divs.append(np.log(delta))

    mean_log_div = np.mean(log_divs, axis=0)

    # Fit in the requested window
    mask = (t_common >= t_min) & (t_common <= t_max)
    if mask.sum() < 3:
        print("Warning: Lyapunov window too narrow; using full range.")
        mask = np.ones(len(t_common), dtype=bool)

    coeffs    = np.polyfit(t_common[mask], mean_log_div[mask], 1)
    lyap      = coeffs[0]
    fit_line  = np.polyval(coeffs, t_common)

    return t_common, mean_log_div, fit_line, lyap, pairs, theta1_interp


# ─────────────────── PLOTTING ─────────────────────────────────────────────────

def make_colormap(n, cmap_name="plasma"):
    cmap   = get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_angles(trials, colors, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Angle vs. Time — All Trials", fontsize=14, fontweight="bold")

    for ax, key, label in zip(axes,
                               ["theta1", "theta2"],
                               [r"$\theta_1$ — Upper Arm (rad)",
                                r"$\theta_2$ — Lower Arm (rad)"]):
        for idx, tr in enumerate(trials):
            name = os.path.basename(tr["path"]).replace(".csv", "")
            ax.plot(tr["t"], tr[key], color=colors[idx],
                    lw=1.2, alpha=0.85, label=name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    path = os.path.join(outdir, "01_angles_vs_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_phase_space(trials, colors, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase-Space Plots — All Trials", fontsize=14, fontweight="bold")

    for ax, (tk, ok, xl, yl, title, wrap) in zip(axes, [
        ("theta1", "omega1", r"$\theta_1$ (rad)", r"$\omega_1$ (rad/s)", "Upper Arm", False),
        ("theta2", "omega2", r"$\theta_2$ (rad)", r"$\omega_2$ (rad/s)", "Lower Arm", True),
    ]):
        for idx, tr in enumerate(trials):
            name = os.path.basename(tr["path"]).replace(".csv", "")
            # For theta2, wrap into (-pi, pi] for display so all trials
            # overlay on the same angular reference regardless of how many
            # full rotations arm 2 completed. This is display-only — the
            # unwrapped theta2 is still used for all other calculations.
            theta_plot = (tr[tk] + np.pi) % (2 * np.pi) - np.pi if wrap else tr[tk]
            ax.plot(theta_plot, tr[ok], color=colors[idx],
                    lw=0.8, alpha=0.75, label=name)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout()
    path = os.path.join(outdir, "02_phase_space.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_divergence(t_common, theta1_interp, pairs, trials, colors, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Trajectory Divergence — |Δθ₁(t)| Between Pairs", fontsize=13)

    pair_colors = get_cmap("tab20")
    for k, (i, j) in enumerate(pairs):
        delta = np.abs(theta1_interp[i] - theta1_interp[j])
        ni = os.path.basename(trials[i]["path"]).replace(".csv", "")
        nj = os.path.basename(trials[j]["path"]).replace(".csv", "")
        ax.plot(t_common, delta, lw=0.9, alpha=0.6,
                color=pair_colors(k / max(len(pairs) - 1, 1)),
                label=f"{ni} vs {nj}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"|Δθ₁| (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    plt.tight_layout()
    path = os.path.join(outdir, "03_divergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_log_divergence(t_common, mean_log_div, fit_line, lyap,
                        t_min, t_max, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Mean Log Divergence & Lyapunov Fit\n"
                 f"λ ≈ {lyap:.3f} s⁻¹  (fit window: {t_min}–{t_max} s)",
                 fontsize=13)

    # Only draw the fit line within the fit window, not extrapolated
    mask_win = (t_common >= t_min) & (t_common <= t_max)

    ax.plot(t_common, mean_log_div, color="#2196F3", lw=1.5,
            label=r"$\langle \ln|\Delta\theta_1|\rangle$")
    ax.plot(t_common[mask_win], fit_line[mask_win], color="#F44336", lw=2.5,
            ls="--", label=f"Linear fit  (λ = {lyap:.3f} s⁻¹)")
    ax.axvspan(t_min, t_max, alpha=0.1, color="red", label="Fit window")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\langle \ln|\Delta\theta_1|\rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "04_lyapunov.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary(trials, colors, t_common, mean_log_div, fit_line, lyap, outdir):
    """One combined figure with the four key panels."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Double Pendulum — Chaos Analysis Summary",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── θ₁(t) ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, tr in enumerate(trials):
        ax1.plot(tr["t"], tr["theta1"], color=colors[idx], lw=1.0, alpha=0.8,
                 label=os.path.basename(tr["path"]).replace(".csv", ""))
    ax1.set_title(r"$\theta_1(t)$ — Upper Arm")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"$\theta_1$ (rad)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2)

    # ── θ₂(t) ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, tr in enumerate(trials):
        ax2.plot(tr["t"], tr["theta2"], color=colors[idx], lw=1.0, alpha=0.8)
    ax2.set_title(r"$\theta_2(t)$ — Lower Arm")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel(r"$\theta_2$ (rad)")
    ax2.grid(True, alpha=0.3)

    # ── Phase space θ₁ ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, tr in enumerate(trials):
        ax3.plot(tr["theta1"], tr["omega1"], color=colors[idx],
                 lw=0.7, alpha=0.7)
    ax3.set_title(r"Phase Space — Upper Arm")
    ax3.set_xlabel(r"$\theta_1$ (rad)")
    ax3.set_ylabel(r"$\omega_1$ (rad/s)")
    ax3.grid(True, alpha=0.3)

    # ── Log divergence + Lyapunov ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    mask_win = (t_common >= LYAP_T_MIN) & (t_common <= LYAP_T_MAX)
    ax4.plot(t_common, mean_log_div, color="#2196F3", lw=1.5,
             label=r"$\langle \ln|\Delta\theta_1|\rangle$")
    ax4.plot(t_common[mask_win], fit_line[mask_win], color="#F44336", lw=2.5,
             ls="--", label=f"Fit  λ = {lyap:.3f} s⁻¹")
    ax4.axvspan(LYAP_T_MIN, LYAP_T_MAX, alpha=0.1, color="red")
    ax4.set_title(f"Lyapunov Exponent Estimate\nλ ≈ {lyap:.3f} s⁻¹")
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel(r"$\langle \ln|\Delta\theta_1|\rangle$")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    path = os.path.join(outdir, "00_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────── MAIN ─────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Find CSV files ────────────────────────────────────────────────────
    files = sorted(glob.glob(CSV_PATTERN))
    if not files:
        print(
            f"\n[ERROR] No files matched '{CSV_PATTERN}'.\n"
            "  → Place your Tracker CSV files in the same folder as this script,\n"
            "    named trial_1.csv, trial_2.csv, … trial_7.csv (or edit CSV_PATTERN).\n"
        )
        return

    print(f"\nFound {len(files)} trial file(s):")
    for f in files:
        print(f"  {f}")

    # ── 2. Process every trial ───────────────────────────────────────────────
    print("\nProcessing trials …")
    trials = []
    for f in files:
        try:
            tr = process_trial(f)
            trials.append(tr)
            n = len(tr["t"])
            dt = np.mean(np.diff(tr["t"]))
            omega_src = "Tracker ω" if tr.get("has_omega") else "numerical diff"
            print(f"  {os.path.basename(f):20s}  {n} frames  "
                  f"Δt≈{dt*1000:.1f} ms  "
                  f"θ₁∈[{np.degrees(tr['theta1'].min()):.1f}°,"
                  f"{np.degrees(tr['theta1'].max()):.1f}°]  "
                  f"[ω source: {omega_src}]")
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")

    if not trials:
        print("[ERROR] All files failed to load. Check your CSV format and column names.")
        return

    # ── 3. Common color palette ──────────────────────────────────────────────
    colors = make_colormap(len(trials))

    # ── 4. Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_angles(trials, colors, OUTPUT_DIR)
    plot_phase_space(trials, colors, OUTPUT_DIR)

    # ── 5. Lyapunov analysis ─────────────────────────────────────────────────
    if len(trials) >= 2:
        t_common, mean_log_div, fit_line, lyap, pairs, theta1_interp = \
            lyapunov_estimate(trials, LYAP_T_MIN, LYAP_T_MAX)

        plot_divergence(t_common, theta1_interp, pairs, trials, colors, OUTPUT_DIR)
        plot_log_divergence(t_common, mean_log_div, fit_line, lyap,
                            LYAP_T_MIN, LYAP_T_MAX, OUTPUT_DIR)
        plot_summary(trials, colors, t_common, mean_log_div, fit_line, lyap, OUTPUT_DIR)

        print(f"\n{'─'*50}")
        print(f"  Estimated Lyapunov exponent  λ ≈ {lyap:.4f} s⁻¹")
        if lyap > 0:
            print(f"  Doubling time              τ₂ ≈ {np.log(2)/lyap:.3f} s")
            print("  → Positive λ confirms chaotic behaviour ✓")
        else:
            print("  → λ ≤ 0: no divergence detected in the chosen window.")
            print("    Try extending LYAP_T_MAX or checking data quality.")
        print(f"{'─'*50}\n")
    else:
        print("Only one trial loaded; skipping Lyapunov / divergence analysis.")

    print(f"All figures saved to: {os.path.abspath(OUTPUT_DIR)}/\n")


if __name__ == "__main__":
    main()
