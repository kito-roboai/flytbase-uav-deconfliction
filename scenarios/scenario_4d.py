"""
Scenario 3 – 4D Extra Credit (3D Space + Time)
================================================
Drones operate at different altitudes (z).  A vertical crossing creates a
conflict only when the drones are at the same altitude at the same time.

Two sub-cases are shown:
  Case A – 3D conflict:  drones cross both in (x,y) AND in z at the same time.
  Case B – 3D clear:     drones cross in (x,y) but at different altitudes → safe.

Expected results: Case A = 'conflict detected', Case B = 'clear'
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DroneMission, Waypoint
from src.deconfliction import check_conflicts
from src.visualization import plot_3d_scenario, animate_4d, plot_distance_vs_time

SAFETY_BUFFER = 8.0   # metres — drones must stay at least 8m apart in 3D space
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# Case A – 3D conflict
# ---------------------------------------------------------------------------

def build_case_a():
    """
    Both drones cross the same point in 3D space at the same time → CONFLICT.

    Primary ALPHA-3D : climbs from z=0 to z=50 while flying East
                       path: (0,0,0) → (100,100,50)
    SIM-X            : descends from z=50 to z=0 while flying North
                       path: (100,0,50) → (0,100,0)

    Crossing point: both drones pass through (50, 50, 25) at t ≈ 5.3s
    → same location, same time → 3D CONFLICT
    """
    primary = DroneMission(
        drone_id="ALPHA-3D",
        waypoints=[
            Waypoint(0,   0,   0),   # starts at ground level, south-west
            Waypoint(100, 100, 50),  # ends at 50m altitude, north-east
        ],
        speed=14.14,          # covers ~154m total distance in ~10.6s
        start_time=0.0,
        mission_end=15.0,
    )

    # SIM-X flies the mirror path — from north-east at altitude 50 down to south-west at 0
    # Both drones move at the same speed → they meet exactly in the middle at t≈5.3s
    sim_x = DroneMission(
        drone_id="SIM-X",
        waypoints=[
            Waypoint(100, 0,  50),   # starts at 50m altitude, south-east
            Waypoint(0,   100, 0),   # ends at ground level, north-west
        ],
        speed=14.14,
        start_time=0.0,
    )

    return primary, [sim_x]


# ---------------------------------------------------------------------------
# Case B – 3D clear (altitude separation)
# ---------------------------------------------------------------------------

def build_case_b():
    """
    Drones cross the same (x,y) point but at DIFFERENT altitudes → CLEAR.

    Primary ALPHA-3D : same as Case A, max altitude = 50m
    SIM-HIGH         : flies at constant altitude z=100m (always 50m above primary)

    Even though their x,y paths cross, the 50m altitude gap keeps them safe.
    This proves the system correctly uses 3D distance, not just 2D.
    """
    primary = DroneMission(
        drone_id="ALPHA-3D",
        waypoints=[
            Waypoint(0,   0,   0),
            Waypoint(100, 100, 50),  # max altitude = 50m
        ],
        speed=14.14,
        start_time=0.0,
        mission_end=15.0,
    )

    # SIM-HIGH stays at z=100 throughout — always at least 50m above primary
    # → 3D distance never drops below safety buffer → CLEAR
    sim_high = DroneMission(
        drone_id="SIM-HIGH",
        waypoints=[
            Waypoint(100, 0,   100),  # constant altitude 100m
            Waypoint(0,   100, 100),  # constant altitude 100m
        ],
        speed=14.14,
        start_time=0.0,
    )

    return primary, [sim_high]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(output_dir: str = OUTPUT_DIR, show: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── Case A: 3D Conflict ───────────────────────────────────────────────
    primary_a, sim_a = build_case_a()
    result_a = check_conflicts(primary_a, sim_a, safety_buffer=SAFETY_BUFFER)

    print("=" * 60)
    print("SCENARIO 3A – 4D Conflict (3D space + time)")
    print("=" * 60)
    print(result_a.summary())

    # 3D static plot — shows crossing paths in x,y,z space with conflict marker
    plot_3d_scenario(
        primary_a, sim_a, result_a,
        title="Scenario 3A – 3D Conflict",
        save_path=os.path.join(output_dir, "scenario_4d_case_a_3d.png"),
        show=show,
    )

    # Distance vs time graph — line dips below buffer at t≈5.3s
    plot_distance_vs_time(
        primary_a, sim_a, result_a,
        safety_buffer=SAFETY_BUFFER,
        title="Scenario 3A – 3D Separation vs Time",
        save_path=os.path.join(output_dir, "scenario_4d_case_a_dist.png"),
        show=show,
    )

    # 4D animation — 3D moving dots with trail, time shown in title
    animate_4d(
        primary_a, sim_a, result_a,
        speed_factor=1.0,
        title="Scenario 3A – 4D Animation (3D+Time) – CONFLICT",
        save_path=os.path.join(output_dir, "scenario_4d_case_a_anim.gif"),
        show=show,
    )

    # ── Case B: 3D Clear ─────────────────────────────────────────────────
    primary_b, sim_b = build_case_b()
    result_b = check_conflicts(primary_b, sim_b, safety_buffer=SAFETY_BUFFER)

    print("=" * 60)
    print("SCENARIO 3B – 4D Clear (altitude separation)")
    print("=" * 60)
    print(result_b.summary())

    # 3D static plot — shows paths never actually meet in 3D space
    plot_3d_scenario(
        primary_b, sim_b, result_b,
        title="Scenario 3B – 3D Clear (altitude separation)",
        save_path=os.path.join(output_dir, "scenario_4d_case_b_3d.png"),
        show=show,
    )

    # 4D animation — drones pass "through" the same x,y but at different altitudes
    animate_4d(
        primary_b, sim_b, result_b,
        speed_factor=1.0,
        title="Scenario 3B – 4D Animation (3D+Time) – CLEAR",
        save_path=os.path.join(output_dir, "scenario_4d_case_b_anim.gif"),
        show=show,
    )

    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    run(show=False)
