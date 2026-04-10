"""
Strategic Deconfliction Engine
================================
Public API
----------
    check_conflicts(primary, simulated, safety_buffer) -> DeconflictionResult

Algorithm
---------
For every pair of flight segments (one from the primary drone, one from a
simulated drone) the engine calls ``closest_approach_on_segment_pair`` from
``trajectory.py``, which analytically minimises the inter-drone distance as a
quadratic function of time.  This gives **exact** spatio-temporal conflict
detection without any discrete time-stepping.

Complexity: O(M × N) segment pairs, where M and N are the number of segments
in the primary and simulated missions respectively.
"""

from __future__ import annotations

from typing import List

from .models import ConflictDetail, DeconflictionResult, DroneMission
from .trajectory import (
    closest_approach_on_segment_pair,
    compute_segment_times,
    mission_duration,
    mission_end_time,
)

# Minimum safe distance between any two drones (in metres)
# If two drones come closer than this → conflict
DEFAULT_SAFETY_BUFFER: float = 5.0


# ---------------------------------------------------------------------------
# Public query interface
# ---------------------------------------------------------------------------

def check_conflicts(
    primary: DroneMission,
    simulated: List[DroneMission],
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
) -> DeconflictionResult:
    """
    Determine whether the primary drone's mission is safe to execute.

    Parameters
    ----------
    primary        : The mission to evaluate.
    simulated      : List of other drones operating in the same airspace.
    safety_buffer  : Minimum allowed separation distance (same units as coords).

    Returns
    -------
    DeconflictionResult with status 'clear' or 'conflict detected', plus full
    conflict details for every violation found.
    """
    # ── Step 1: Feasibility check ─────────────────────────────────────────
    # Can the primary drone finish its mission before the deadline?
    # Example: mission needs 14 seconds but window is only 10 seconds → infeasible
    duration = mission_duration(primary)       # how long mission actually takes
    t_end_actual = mission_end_time(primary)   # absolute time when mission finishes

    feasible = True
    if primary.mission_end is not None:
        # Allow tiny floating point error (1e-9) when comparing times
        feasible = t_end_actual <= primary.mission_end + 1e-9

    # ── Step 2: Get the primary drone's segment time intervals ────────────
    # Example: [(0.0, 5.0), (5.0, 10.0)] → two segments, each 5 seconds
    primary_seg_times = compute_segment_times(primary)

    # The time window we care about for the primary drone
    t_mission_start = primary_seg_times[0][0]   # when primary departs
    t_mission_end = (
        primary.mission_end                      # use deadline if set
        if primary.mission_end is not None
        else primary_seg_times[-1][1]            # otherwise use actual end time
    )

    # ── Step 3: Check every simulated drone for conflicts ─────────────────
    conflicts: List[ConflictDetail] = []

    for sim_drone in simulated:
        sim_seg_times = compute_segment_times(sim_drone)
        # Check all segment combinations between primary and this simulated drone
        _check_drone_pair(
            primary=primary,
            primary_seg_times=primary_seg_times,
            t_mission_start=t_mission_start,
            t_mission_end=t_mission_end,
            sim_drone=sim_drone,
            sim_seg_times=sim_seg_times,
            safety_buffer=safety_buffer,
            conflicts=conflicts,  # conflicts list is filled in-place
        )

    # ── Step 4: Build and return the final result ─────────────────────────
    status = "conflict detected" if conflicts else "clear"
    return DeconflictionResult(
        status=status,
        conflicts=conflicts,
        feasible=feasible,
        actual_duration=duration,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_drone_pair(
    primary,
    primary_seg_times,
    t_mission_start,
    t_mission_end,
    sim_drone,
    sim_seg_times,
    safety_buffer,
    conflicts,
) -> None:
    """
    Check all segment-pair combinations between the primary drone and one
    simulated drone.  Appends ConflictDetail objects to *conflicts* in-place.
    """
    # Number of segments = number of waypoints - 1
    # Example: 3 waypoints → 2 segments (A→B and B→C)
    n_primary_segs = len(primary.waypoints) - 1
    n_sim_segs     = len(sim_drone.waypoints) - 1

    for i in range(n_primary_segs):
        pt_start, pt_end = primary_seg_times[i]  # time window of this primary segment

        # Skip primary segments that are completely outside the mission window
        # (happens when mission_end cuts off part of the path)
        if pt_end < t_mission_start or pt_start > t_mission_end:
            continue

        # Clamp segment times to the allowed mission window
        pt_start_eff = max(pt_start, t_mission_start)
        pt_end_eff   = min(pt_end,   t_mission_end)

        # 3D positions of this primary segment's start and end
        p1 = primary.waypoints[i].to_array()
        p2 = primary.waypoints[i + 1].to_array()

        for j in range(n_sim_segs):
            st_start, st_end = sim_seg_times[j]  # time window of this sim segment

            # Quick filter: skip if the two segments have NO time overlap at all
            # This saves time — no need to do heavy math if they fly at different times
            if st_end < pt_start_eff or st_start > pt_end_eff:
                continue

            # 3D positions of this simulated segment's start and end
            q1 = sim_drone.waypoints[j].to_array()
            q2 = sim_drone.waypoints[j + 1].to_array()

            # Core math: find the exact time and distance of closest approach
            # Returns: (time, primary_position, sim_position, minimum_distance)
            t_closest, pos_p, pos_q, min_dist = closest_approach_on_segment_pair(
                p1, p2, pt_start_eff, pt_end_eff,
                q1, q2, st_start, st_end,
            )

            # If drones came closer than the safety buffer → CONFLICT
            if min_dist < safety_buffer:
                conflicts.append(
                    ConflictDetail(
                        conflicting_drone_id=sim_drone.drone_id,
                        time=t_closest,
                        location=(float(pos_p[0]), float(pos_p[1]), float(pos_p[2])),
                        separation=min_dist,       # actual distance (less than buffer)
                        safety_buffer=safety_buffer,
                    )
                )
