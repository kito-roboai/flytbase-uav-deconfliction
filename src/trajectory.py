"""
Continuous Trajectory Mathematics
==================================
Every drone moves at a *constant speed* along an ordered list of waypoints.
There are NO pre-assigned timestamps for intermediate waypoints — they are
derived analytically from the segment lengths and the drone's speed.

Key guarantee: positions are computed as a **continuous function of time**,
never by stepping through discrete time samples.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .models import DroneMission, Waypoint


# ---------------------------------------------------------------------------
# Segment timing
# ---------------------------------------------------------------------------

def compute_segment_times(mission: DroneMission) -> List[Tuple[float, float]]:
    """
    Analytically compute the arrival time at each waypoint and return the
    time interval ``[t_start, t_end]`` for every flight segment.

    The drone departs the first waypoint at ``mission.start_time`` and
    travels each segment at ``mission.speed``.  No discrete sampling is used.

    Returns
    -------
    List of (t_start, t_end) tuples, one per segment.
    """
    # Start with the mission's departure time
    times: List[float] = [mission.start_time]

    for i in range(len(mission.waypoints) - 1):
        p1 = mission.waypoints[i].to_array()      # current waypoint position
        p2 = mission.waypoints[i + 1].to_array()  # next waypoint position

        # Distance between two waypoints (straight line in 3D)
        # Formula: sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
        segment_length = float(np.linalg.norm(p2 - p1))

        # Time = Distance / Speed  (basic physics formula)
        # If two waypoints are the same point → distance=0 → time=0 (no crash)
        dt = segment_length / mission.speed if segment_length > 0 else 0.0

        # Each segment ends when previous segment ended + travel time
        times.append(times[-1] + dt)

    # Convert flat time list into (start, end) pairs — one per segment
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def mission_duration(mission: DroneMission) -> float:
    """Return the total time (seconds) required to fly the full mission."""
    seg_times = compute_segment_times(mission)
    # Last segment's end time minus first segment's start time
    return seg_times[-1][1] - seg_times[0][0]


def mission_end_time(mission: DroneMission) -> float:
    """Return the absolute time at which the drone reaches its final waypoint."""
    return compute_segment_times(mission)[-1][1]


# ---------------------------------------------------------------------------
# Continuous position query
# ---------------------------------------------------------------------------

def position_at_time(mission: DroneMission, t: float) -> Optional[np.ndarray]:
    """
    Return the drone's 3-D position at time *t* using **closed-form linear
    interpolation** along the appropriate segment.

    Parameters
    ----------
    mission : DroneMission
    t       : Query time in seconds.

    Returns
    -------
    np.ndarray of shape (3,), or ``None`` if *t* is outside the mission window.
    """
    seg_times = compute_segment_times(mission)
    t_mission_start = seg_times[0][0]
    t_mission_end   = seg_times[-1][1]

    # If asked for time before drone departs or after it lands → no position
    if t < t_mission_start or t > t_mission_end:
        return None

    for i, (t_start, t_end) in enumerate(seg_times):
        if t_start <= t <= t_end:
            p1 = mission.waypoints[i].to_array()      # segment start position
            p2 = mission.waypoints[i + 1].to_array()  # segment end position

            dt = t_end - t_start

            # alpha = how far through this segment are we? (0.0 = start, 1.0 = end)
            # Example: t=7, t_start=5, t_end=10 → alpha = (7-5)/(10-5) = 0.4
            alpha = (t - t_start) / dt if dt > 1e-12 else 0.0

            # Linear interpolation: position = start + alpha × (end - start)
            return p1 + alpha * (p2 - p1)

    # Floating-point edge case: t == t_mission_end lands here in rare cases.
    return mission.waypoints[-1].to_array()


def sample_trajectory(
    mission: DroneMission, n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the continuous trajectory at *n_points* evenly-spaced times for
    plotting purposes only.  The deconfliction engine never calls this.

    Returns
    -------
    times     : np.ndarray of shape (n_points,)
    positions : np.ndarray of shape (n_points, 3)
    """
    seg_times = compute_segment_times(mission)
    t0 = seg_times[0][0]   # mission start time
    t1 = seg_times[-1][1]  # mission end time

    # Evenly spaced time samples from start to end
    times = np.linspace(t0, t1, n_points)

    # Get position at each sampled time
    positions = np.array([position_at_time(mission, t) for t in times])
    return times, positions


# ---------------------------------------------------------------------------
# Segment-pair geometry used by the deconfliction engine
# ---------------------------------------------------------------------------

def closest_approach_on_segment_pair(
    p1_start: np.ndarray,
    p1_end: np.ndarray,
    t1_start: float,
    t1_end: float,
    p2_start: np.ndarray,
    p2_end: np.ndarray,
    t2_start: float,
    t2_end: float,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Analytically find the time of **minimum separation** between two drones
    while each is flying along one of its linear segments.

    Both drones move at constant velocity within their respective segments,
    so their *relative displacement* D(t) = P(t) – Q(t) is **linear** in t
    and |D(t)|² is a **quadratic** in t.  The minimum is found in O(1) time
    without any discrete sampling.

    Parameters
    ----------
    p1_start/p1_end   : Start/end positions of the primary drone's segment.
    t1_start/t1_end   : Time interval of the primary drone's segment.
    p2_start/p2_end   : Start/end positions of the simulated drone's segment.
    t2_start/t2_end   : Time interval of the simulated drone's segment.

    Returns
    -------
    t_closest   : Time at closest approach (within temporal overlap).
    pos_primary : Primary drone position at t_closest.
    pos_sim     : Simulated drone position at t_closest.
    min_dist    : Euclidean separation at t_closest.
    """
    # ── Step 1: Find the time window where BOTH drones are flying ─────────
    # Conflict is only possible when both drones are in the air at the same time
    t_lo = max(t1_start, t2_start)  # overlap starts at the later departure
    t_hi = min(t1_end,   t2_end)    # overlap ends at the earlier landing

    if t_lo >= t_hi:
        # No temporal overlap → these two segments can never conflict
        # Return infinity distance as a signal to skip this pair
        pos1 = p1_start.copy()
        pos2 = p2_start.copy()
        return t_lo, pos1, pos2, float("inf")

    # ── Step 2: Express both drones' positions as linear functions of time ─
    # P(t) = p1_start + v1 × (t – t1_start)   → drone 1 position at time t
    # Q(t) = p2_start + v2 × (t – t2_start)   → drone 2 position at time t
    dt1 = t1_end - t1_start
    dt2 = t2_end - t2_start
    v1 = (p1_end - p1_start) / dt1 if dt1 > 1e-12 else np.zeros(3)  # velocity of drone 1
    v2 = (p2_end - p2_start) / dt2 if dt2 > 1e-12 else np.zeros(3)  # velocity of drone 2

    # ── Step 3: Build the relative displacement function D(t) = A + B×t ───
    # D(t) = P(t) - Q(t) = [p1_start - v1×t1_start] - [p2_start - v2×t2_start] + (v1-v2)×t
    A = (p1_start - v1 * t1_start) - (p2_start - v2 * t2_start)  # constant part
    B = v1 - v2                                                     # coefficient of t (relative velocity)

    # ── Step 4: |D(t)|² = a×t² + b×t + c  (quadratic = parabola shape) ───
    # Minimum of this parabola = closest approach point
    a_coef = float(np.dot(B, B))        # coefficient of t²
    b_coef = 2.0 * float(np.dot(A, B)) # coefficient of t
    # c_coef = float(np.dot(A, A))      # constant term (not needed explicitly)

    # ── Step 5: Find minimum of the quadratic over [t_lo, t_hi] ──────────
    # Candidates: the two endpoints + the vertex (if it falls inside the range)
    candidates = [t_lo, t_hi]

    if a_coef > 1e-12:
        # Vertex of parabola = -b / (2a) → this is where minimum distance occurs
        t_vertex = -b_coef / (2.0 * a_coef)
        if t_lo < t_vertex < t_hi:
            candidates.append(t_vertex)  # only use vertex if it's inside the overlap window

    # Helper: squared distance at time t (cheaper than computing sqrt every time)
    def dist_sq(t: float) -> float:
        d = A + B * t
        return float(np.dot(d, d))

    # Pick the candidate time that gives the smallest distance
    t_best = min(candidates, key=dist_sq)

    # Actual distance (take sqrt, clamp to 0 to avoid floating point negatives)
    min_dist = float(np.sqrt(max(0.0, dist_sq(t_best))))

    # Compute the actual 3D positions of both drones at the closest approach time
    pos_primary = p1_start + v1 * (t_best - t1_start)
    pos_sim     = p2_start + v2 * (t_best - t2_start)

    return t_best, pos_primary, pos_sim, min_dist
