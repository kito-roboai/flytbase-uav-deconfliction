"""
Scenario 2 – Conflict Detected
================================
Implements the **sample test case** from the v1.1 spec:

    "Two drones with crossing paths but different velocities — your system must
     determine whether they will occupy the same space at the same time.  Both
     drones start from different positions at different times."

Two additional simulated drones are added to demonstrate multiple simultaneous
conflicts.

Expected result: status = 'conflict detected'
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DroneMission, Waypoint
from src.deconfliction import check_conflicts
from src.visualization import plot_2d_scenario, plot_distance_vs_time, animate_2d

SAFETY_BUFFER = 8.0   # metres — drones must stay at least 8m apart
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def build_scenario():
    """
    Build and return (primary, simulated_list) for the conflict scenario.

    Primary drone flies West → East along y=50 (horizontal line).
    Three simulated drones are placed to create different conflict situations:

    Manual verification of conflict times:
        ALPHA   : (0,50) → (100,50)  speed=10  → reaches x=50 at t=5.0s
        SIM-A   : (50,0) → (50,100)  speed=10  → reaches y=50 at t=5.0s  ← EXACT collision
        SIM-B   : (50,0) → (50,100)  speed=20  → reaches y=50 at t=4.5s  ← near-miss (sep=4.47m < 8m)
        SIM-C   : (0,50) → (100,50)  speed=10  → 0.5s behind ALPHA → gap=5m < 8m  ← tailgating conflict
    """

    # Primary: flies along y=50, west → east, 100 metres in 10 seconds
    primary = DroneMission(
        drone_id="ALPHA",
        waypoints=[
            Waypoint(0, 50),
            Waypoint(100, 50),
        ],
        speed=10.0,
        start_time=0.0,
        mission_end=15.0,   # must finish within 15 seconds
    )

    # SIM-A: flies south → north, crosses ALPHA's path at (50, 50)
    # Same speed (10 m/s), same start time (t=0)
    # → ALPHA reaches x=50 at t=5s, SIM-A reaches y=50 at t=5s
    # → Both at (50,50) at exactly t=5s → distance = 0m → CONFLICT
    sim_a = DroneMission(
        drone_id="SIM-A",
        waypoints=[
            Waypoint(50, 0),
            Waypoint(50, 100),
        ],
        speed=10.0,
        start_time=0.0,
    )

    # SIM-B: same path as SIM-A but 2x faster and starts 2 seconds late
    # → reaches y=50 at t = 2 + (50/20) = 4.5s
    # → at t=4.5s: ALPHA is at x=45, distance from (50,50) = sqrt(5²+0²) = 5m
    # → closest approach ≈ 4.47m < 8m buffer → CONFLICT (near-miss)
    sim_b = DroneMission(
        drone_id="SIM-B",
        waypoints=[
            Waypoint(50, 0),
            Waypoint(50, 100),
        ],
        speed=20.0,
        start_time=2.0,     # starts 2 seconds after ALPHA
    )

    # SIM-C: flies the EXACT same path as ALPHA (same lane, same direction)
    # Starts 0.5 seconds later → always 5m behind ALPHA (gap = speed × delay = 10 × 0.5 = 5m)
    # 5m gap < 8m safety buffer → they are too close the entire mission → CONFLICT
    sim_c = DroneMission(
        drone_id="SIM-C",
        waypoints=[
            Waypoint(0, 50),
            Waypoint(100, 50),
        ],
        speed=10.0,
        start_time=0.5,     # 0.5s behind → 5m gap → violates 8m buffer → CONFLICT
    )

    return primary, [sim_a, sim_b, sim_c]


def run(output_dir: str = OUTPUT_DIR, show: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    primary, simulated = build_scenario()

    # Run the deconfliction check — expect CONFLICT DETECTED with 3 conflicts
    result = check_conflicts(primary, simulated, safety_buffer=SAFETY_BUFFER)

    print("=" * 60)
    print("SCENARIO 2 – Conflict Detected")
    print("=" * 60)
    print(result.summary())

    # Save top-down 2D path overview with red X conflict markers
    plot_2d_scenario(
        primary, simulated, result,
        title="Scenario 2 – Conflict Detected",
        save_path=os.path.join(output_dir, "scenario_conflict_2d.png"),
        show=show,
    )

    # Save distance-vs-time graph (lines dip below the red buffer line at conflict times)
    plot_distance_vs_time(
        primary, simulated, result,
        safety_buffer=SAFETY_BUFFER,
        title="Scenario 2 – Separation vs Time",
        save_path=os.path.join(output_dir, "scenario_conflict_dist.png"),
        show=show,
    )

    # Save animated GIF showing drones moving and nearly colliding
    animate_2d(
        primary, simulated, result,
        speed_factor=2.0,
        title="Scenario 2 – Conflict Animation",
        save_path=os.path.join(output_dir, "scenario_conflict_anim.gif"),
        show=show,
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    run(show=False)
