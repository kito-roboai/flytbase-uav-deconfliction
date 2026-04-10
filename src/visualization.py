"""
Visualization Module
====================
Produces static plots and animated GIFs for:
  - 2-D trajectory overview with conflict markers
  - Time-space diagram (distance vs time)
  - 3-D trajectory plot (extra-credit)
  - 4-D animated GIF: 3-D space evolving over time (extra-credit)

All functions accept a save_path argument; if None the figure is shown
interactively.  Set show=False when running headless (e.g. in tests).
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")          # headless-safe backend — no screen needed, saves files directly
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from .models import ConflictDetail, DeconflictionResult, DroneMission
from .trajectory import compute_segment_times, sample_trajectory

# ── Colour palette ────────────────────────────────────────────────────────
PRIMARY_COLOR   = "#1f77b4"   # blue  → primary drone
SIM_COLORS      = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]  # one color per simulated drone
CONFLICT_COLOR  = "#ff0000"   # red   → conflict markers and safety circles
SAFETY_ALPHA    = 0.15        # transparency of the red safety circle fill


# ---------------------------------------------------------------------------
# 2-D static overview
# ---------------------------------------------------------------------------

def plot_2d_scenario(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    title: str = "UAV Deconfliction – 2D Overview",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot all trajectories in the x-y plane with conflict markers.
    Saves a top-down 2D bird's-eye view as a PNG image.

    Returns the save path used (or empty string if shown interactively).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")   # keep x and y scale equal so paths look correct
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Draw primary drone path (solid blue line) ─────────────────────────
    _, pts_p = sample_trajectory(primary, n_points=300)
    ax.plot(pts_p[:, 0], pts_p[:, 1],
            color=PRIMARY_COLOR, linewidth=2.5, label=f"Primary ({primary.drone_id})", zorder=3)
    _plot_waypoints(ax, primary, PRIMARY_COLOR, marker="D", zorder=4)  # diamond markers at waypoints

    # ── Draw each simulated drone path (dashed colored lines) ────────────
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]  # cycle through colors if many drones
        _, pts_s = sample_trajectory(sim, n_points=300)
        ax.plot(pts_s[:, 0], pts_s[:, 1],
                color=color, linewidth=1.8, linestyle="--",
                label=f"Sim {sim.drone_id}", zorder=2)
        _plot_waypoints(ax, sim, color, marker="s", zorder=3)  # square markers at waypoints

    # ── Mark each conflict location with a red X and safety circle ────────
    if result.conflicts:
        for c in result.conflicts:
            # Red X at conflict point
            ax.scatter(c.location[0], c.location[1],
                       s=200, color=CONFLICT_COLOR, marker="X", zorder=5)
            # Transparent red circle showing safety buffer zone
            circle = plt.Circle(
                (c.location[0], c.location[1]),
                c.safety_buffer,
                color=CONFLICT_COLOR, alpha=SAFETY_ALPHA, zorder=1,
            )
            ax.add_patch(circle)
            # Label showing time and separation distance
            ax.annotate(
                f"  t={c.time:.1f}s\n  Δ={c.separation:.1f}m",
                xy=(c.location[0], c.location[1]),
                fontsize=7, color=CONFLICT_COLOR,
            )

    # ── Status banner at the bottom of the image ─────────────────────────
    status_color = "#d62728" if not result.is_clear else "#2ca02c"  # red if conflict, green if clear
    ax.set_facecolor("#f9f9f9")
    fig.text(0.5, 0.01, f"STATUS: {result.status.upper()}",
             ha="center", fontsize=12, fontweight="bold", color=status_color)

    handles, labels = ax.get_legend_handles_labels()
    if result.conflicts:
        handles.append(mpatches.Patch(color=CONFLICT_COLOR, label="Conflict zone"))
    ax.legend(handles=handles + [], loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# Distance-vs-time diagram
# ---------------------------------------------------------------------------

def plot_distance_vs_time(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    safety_buffer: float = 5.0,
    title: str = "Inter-Drone Separation vs Time",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot separation distance between primary and each simulated drone over time.
    The red dashed line shows the safety buffer — anything below it is a conflict.
    """
    from .trajectory import position_at_time

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Separation (m)")
    ax.grid(True, linestyle="--", alpha=0.5)

    seg_times_p = compute_segment_times(primary)
    t0_p = seg_times_p[0][0]   # primary mission start time
    t1_p = seg_times_p[-1][1]  # primary mission end time

    for k, sim in enumerate(simulated):
        seg_times_s = compute_segment_times(sim)
        t0_s = seg_times_s[0][0]
        t1_s = seg_times_s[-1][1]

        # Only plot during the time both drones are flying
        t_lo = max(t0_p, t0_s)
        t_hi = min(t1_p, t1_s)
        if t_lo >= t_hi:
            continue  # no overlap in time → nothing to plot for this drone

        # Sample 500 time points and compute distance at each
        t_vals = np.linspace(t_lo, t_hi, 500)
        dists = []
        for t in t_vals:
            pp = position_at_time(primary, t)
            ps = position_at_time(sim, t)
            if pp is not None and ps is not None:
                dists.append(float(np.linalg.norm(pp - ps)))  # Euclidean distance
            else:
                dists.append(float("nan"))  # gap in line if one drone not flying

        color = SIM_COLORS[k % len(SIM_COLORS)]
        ax.plot(t_vals, dists, color=color, linewidth=1.8, label=f"Sim {sim.drone_id}")

    # Red dashed line = safety buffer threshold
    ax.axhline(safety_buffer, color=CONFLICT_COLOR, linestyle="--",
               linewidth=1.5, label=f"Safety buffer ({safety_buffer}m)")
    # Light red shaded zone below the buffer (danger zone)
    ax.fill_between([t0_p, t1_p], 0, safety_buffer,
                    color=CONFLICT_COLOR, alpha=0.07)

    # Vertical dotted lines at each conflict time
    for c in result.conflicts:
        ax.axvline(c.time, color=CONFLICT_COLOR, linestyle=":", alpha=0.7)

    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 2-D animation
# ---------------------------------------------------------------------------

def animate_2d(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    fps: int = 20,
    speed_factor: float = 1.0,
    title: str = "UAV Deconfliction – Animation",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Produce an animated GIF showing all drones moving over time.
    Primary drone = blue circle, simulated drones = colored squares.
    Conflict spots are marked with static red X markers.

    Parameters
    ----------
    fps           : Frames per second of the output animation.
    speed_factor  : Playback speed multiplier (>1 = faster than real-time).
    """
    from .trajectory import position_at_time

    # Find global start and end times across all drones
    seg_times_p = compute_segment_times(primary)
    t0_global = min(seg_times_p[0][0],
                    *(compute_segment_times(s)[0][0] for s in simulated))
    t1_global = max(seg_times_p[-1][1],
                    *(compute_segment_times(s)[-1][1] for s in simulated))

    # Total number of animation frames
    n_frames = max(int(fps * (t1_global - t0_global) / speed_factor), 10)
    times = np.linspace(t0_global, t1_global, n_frames)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Draw faint background paths so viewer can see full routes
    _, pts_p = sample_trajectory(primary, n_points=300)
    ax.plot(pts_p[:, 0], pts_p[:, 1], color=PRIMARY_COLOR,
            linewidth=1.0, alpha=0.3)
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        _, pts_s = sample_trajectory(sim, n_points=300)
        ax.plot(pts_s[:, 0], pts_s[:, 1], color=color,
                linewidth=1.0, alpha=0.3, linestyle="--")

    # Static red X markers at all conflict locations
    for c in result.conflicts:
        ax.scatter(c.location[0], c.location[1],
                   s=150, color=CONFLICT_COLOR, marker="X", zorder=5, alpha=0.7)

    # Moving drone dots — these update every frame
    drone_dot_p, = ax.plot([], [], "o", color=PRIMARY_COLOR,
                           markersize=12, label=f"Primary ({primary.drone_id})", zorder=6)
    sim_dots = []
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        dot, = ax.plot([], [], "s", color=color,
                       markersize=9, label=f"Sim {sim.drone_id}", zorder=5)
        sim_dots.append((sim, dot))

    # Time counter text in top-left corner
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                        fontsize=9, verticalalignment="top")
    ax.legend(fontsize=7, loc="upper right")

    # Set axis limits with a margin so drones don't touch the edges
    all_pts = [pts_p]
    for sim in simulated:
        _, pts_s = sample_trajectory(sim, n_points=100)
        all_pts.append(pts_s)
    all_xy = np.vstack(all_pts)
    margin = 10.0
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    def _init():
        # Clear all moving elements at the start of animation
        drone_dot_p.set_data([], [])
        for _, dot in sim_dots:
            dot.set_data([], [])
        time_text.set_text("")
        return [drone_dot_p] + [d for _, d in sim_dots] + [time_text]

    def _update(frame_idx):
        # Called once per frame — updates drone positions and time label
        t = times[frame_idx]

        # Move primary drone dot to its position at time t
        pos_p = position_at_time(primary, t)
        if pos_p is not None:
            drone_dot_p.set_data([pos_p[0]], [pos_p[1]])
        else:
            drone_dot_p.set_data([], [])  # hide dot if drone not flying yet

        # Move each simulated drone dot
        for sim, dot in sim_dots:
            pos_s = position_at_time(sim, t)
            if pos_s is not None:
                dot.set_data([pos_s[0]], [pos_s[1]])
            else:
                dot.set_data([], [])

        time_text.set_text(f"t = {t:.1f} s")
        return [drone_dot_p] + [d for _, d in sim_dots] + [time_text]

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init,
        frames=n_frames, interval=1000 // fps, blit=True,
    )

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"  # pillow for GIF, ffmpeg for MP4
        anim.save(save_path, writer=writer, fps=fps)
        plt.close(fig)
        return save_path
    if show:
        plt.show()
    plt.close(fig)
    return ""


# ---------------------------------------------------------------------------
# 3-D static plot
# ---------------------------------------------------------------------------

def plot_3d_scenario(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    title: str = "UAV Deconfliction – 3D Overview",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot trajectories in 3-D space (x, y, z).
    Same as 2D plot but with altitude (z) shown on the third axis.
    Used for Scenario 3 where drones fly at different altitudes.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")  # 3D axes
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z / Alt (m)")

    # Primary drone path in 3D
    _, pts_p = sample_trajectory(primary, n_points=300)
    ax.plot(pts_p[:, 0], pts_p[:, 1], pts_p[:, 2],
            color=PRIMARY_COLOR, linewidth=2.5, label=f"Primary ({primary.drone_id})")
    wpts = np.array([wp.to_array() for wp in primary.waypoints])
    ax.scatter(wpts[:, 0], wpts[:, 1], wpts[:, 2],
               color=PRIMARY_COLOR, marker="D", s=60, zorder=5)

    # Simulated drone paths in 3D
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        _, pts_s = sample_trajectory(sim, n_points=300)
        ax.plot(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                color=color, linewidth=1.8, linestyle="--", label=f"Sim {sim.drone_id}")
        wpts_s = np.array([wp.to_array() for wp in sim.waypoints])
        ax.scatter(wpts_s[:, 0], wpts_s[:, 1], wpts_s[:, 2],
                   color=color, marker="s", s=40, zorder=4)

    # Red X markers at conflict locations (in 3D space)
    for c in result.conflicts:
        ax.scatter(*c.location, color=CONFLICT_COLOR, marker="X", s=200, zorder=6)

    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 4-D animated plot: 3-D space + time 
# ---------------------------------------------------------------------------

def animate_4d(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    fps: int = 15,
    speed_factor: float = 1.0,
    title: str = "UAV Deconfliction – 4D (3D + Time)",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Animate all drone positions in 3-D space over time.
    Each drone has a moving dot + a trail showing recent positions.
    The title updates with current time so viewer can track when conflicts happen.
    """
    from .trajectory import position_at_time

    # Global time range across all drones
    seg_times_p = compute_segment_times(primary)
    t0_global = min(seg_times_p[0][0],
                    *(compute_segment_times(s)[0][0] for s in simulated))
    t1_global = max(seg_times_p[-1][1],
                    *(compute_segment_times(s)[-1][1] for s in simulated))

    n_frames = max(int(fps * (t1_global - t0_global) / speed_factor), 10)
    times = np.linspace(t0_global, t1_global, n_frames)
    trail_len = max(n_frames // 10, 5)  # how many past positions to show as trail

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z / Alt (m)")

    # Set fixed axis limits so the view doesn't jump around during animation
    all_wpts = [wp.to_array() for wp in primary.waypoints]
    for sim in simulated:
        all_wpts += [wp.to_array() for wp in sim.waypoints]
    all_wpts = np.array(all_wpts)
    margin = 10.0
    ax.set_xlim(all_wpts[:, 0].min() - margin, all_wpts[:, 0].max() + margin)
    ax.set_ylim(all_wpts[:, 1].min() - margin, all_wpts[:, 1].max() + margin)
    ax.set_zlim(all_wpts[:, 2].min() - margin, all_wpts[:, 2].max() + margin)

    # Faint background paths (full routes shown at low opacity)
    _, pts_p = sample_trajectory(primary, n_points=300)
    ax.plot(pts_p[:, 0], pts_p[:, 1], pts_p[:, 2],
            color=PRIMARY_COLOR, linewidth=1.0, alpha=0.25)
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        _, pts_s = sample_trajectory(sim, n_points=300)
        ax.plot(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                color=color, linewidth=1.0, alpha=0.25, linestyle="--")

    # Static red X markers at conflict locations
    for c in result.conflicts:
        ax.scatter(*c.location, color=CONFLICT_COLOR, marker="X", s=200, zorder=6)

    # History buffers — store recent positions to draw the trail
    hist_p: list = []
    hists_s: list = [[] for _ in simulated]

    # Moving dot + trail line for primary drone
    dot_p, = ax.plot([], [], [], "o", color=PRIMARY_COLOR, markersize=10, zorder=7,
                     label=f"Primary ({primary.drone_id})")
    trail_p, = ax.plot([], [], [], "-", color=PRIMARY_COLOR, linewidth=2.0, alpha=0.5)

    # Moving dot + trail line for each simulated drone
    sim_artists = []
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        dot, = ax.plot([], [], [], "s", color=color, markersize=7, zorder=6,
                       label=f"Sim {sim.drone_id}")
        trail, = ax.plot([], [], [], "-", color=color, linewidth=1.5, alpha=0.5)
        sim_artists.append((sim, dot, trail))

    ax.legend(fontsize=7, loc="upper left")

    def _update(frame_idx):
        t = times[frame_idx]
        # Update title with current time so viewer knows when conflicts happen
        ax.set_title(f"{title}\nt = {t:.1f} s", fontsize=11, fontweight="bold")

        # Update primary drone dot and trail
        pos_p = position_at_time(primary, t)
        if pos_p is not None:
            hist_p.append(pos_p.copy())
            if len(hist_p) > trail_len:
                hist_p.pop(0)  # remove oldest position to keep trail length fixed
            trail_arr = np.array(hist_p)
            dot_p.set_data_3d([pos_p[0]], [pos_p[1]], [pos_p[2]])
            trail_p.set_data_3d(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2])

        # Update each simulated drone dot and trail
        for idx, (sim, dot, trail) in enumerate(sim_artists):
            pos_s = position_at_time(sim, t)
            if pos_s is not None:
                hists_s[idx].append(pos_s.copy())
                if len(hists_s[idx]) > trail_len:
                    hists_s[idx].pop(0)
                tr = np.array(hists_s[idx])
                dot.set_data_3d([pos_s[0]], [pos_s[1]], [pos_s[2]])
                trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        artists = [dot_p, trail_p] + [a for _, a, t_ in sim_artists] + [t_ for _, _, t_ in sim_artists]
        return artists

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps)
        plt.close(fig)
        return save_path
    if show:
        plt.show()
    plt.close(fig)
    return ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_waypoints(ax, mission: DroneMission, color: str,
                    marker: str = "o", zorder: int = 3) -> None:
    """Plot a dot at each waypoint and label it W0, W1, W2 etc."""
    for i, wp in enumerate(mission.waypoints):
        ax.scatter(wp.x, wp.y, color=color, marker=marker, s=50, zorder=zorder)
        ax.annotate(f"W{i}", (wp.x, wp.y), fontsize=6,
                    color=color, xytext=(3, 3), textcoords="offset points")


def _save_or_show(fig, save_path: Optional[str], show: bool) -> str:
    """Save the figure to a file or display it on screen."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    if show:
        plt.show()
    plt.close(fig)
    return ""
