"""
Scenario 1 – Conflict-Free Mission
====================================
Two simulated drones fly paths that are spatially near the primary drone's
route but are *temporally offset* so their closest approaches never violate
the safety buffer.

Expected result: status = 'clear'
"""

from __future__ import annotations

import os
import sys

# Allow running as a standalone script from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DroneMission, Waypoint
from src.deconfliction import check_conflicts
from src.visualization import plot_2d_scenario, plot_distance_vs_time, animate_2d

SAFETY_BUFFER = 10.0  # metres — drones must stay at least 10m apart
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def build_scenario():
    """
    Build and return (primary, simulated_list) for the conflict-free scenario.

    Key idea: The simulated drones are near ALPHA's path in space,
    but they fly at completely different TIMES — so they never actually
    share the same location at the same moment.
    """

    # Primary drone ALPHA: flies diagonally from bottom-left to top-right
    # Path: (0,0) → (50,50) → (100,100)
    # Speed: 10 m/s, departs at t=0
    # Must finish by t=30 (mission_end=30)
    primary = DroneMission(
        drone_id="ALPHA",
        waypoints=[
            Waypoint(0, 0),
            Waypoint(50, 50),
            Waypoint(100, 100),
        ],
        speed=10.0,
        start_time=0.0,
        mission_end=30.0,
    )

    # SIM-1: flies a parallel path shifted 30m to the right
    # BUT departs at t=-20 (20 seconds BEFORE ALPHA)
    # → SIM-1 is already far ahead (or gone) by the time ALPHA is flying
    # → No conflict even though the paths are close in space
    sim1 = DroneMission(
        drone_id="SIM-1",
        waypoints=[
            Waypoint(30, 0),
            Waypoint(80, 50),
            Waypoint(130, 100),
        ],
        speed=15.0,
        start_time=-20.0,   # departed 20 seconds before ALPHA → already gone
    )

    # SIM-2: crosses ALPHA's path diagonally (from top-left to bottom-right)
    # BUT departs at t=50, long after ALPHA finishes at ~t=14
    # → No conflict because they fly at completely different times
    sim2 = DroneMission(
        drone_id="SIM-2",
        waypoints=[
            Waypoint(0, 100),
            Waypoint(100, 0),
        ],
        speed=20.0,
        start_time=50.0,    # departs long after ALPHA has already landed
    )

    return primary, [sim1, sim2]


def run(output_dir: str = OUTPUT_DIR, show: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    primary, simulated = build_scenario()

    # Run the deconfliction check — expect CLEAR result
    result = check_conflicts(primary, simulated, safety_buffer=SAFETY_BUFFER)

    print("=" * 60)
    print("SCENARIO 1 – Conflict-Free Mission")
    print("=" * 60)
    print(result.summary())

    # Save top-down 2D path overview as PNG
    plot_2d_scenario(
        primary, simulated, result,
        title="Scenario 1 – Conflict-Free Mission",
        save_path=os.path.join(output_dir, "scenario_clear_2d.png"),
        show=show,
    )

    # Save distance-vs-time graph as PNG (should stay above buffer line throughout)
    plot_distance_vs_time(
        primary, simulated, result,
        safety_buffer=SAFETY_BUFFER,
        title="Scenario 1 – Separation vs Time",
        save_path=os.path.join(output_dir, "scenario_clear_dist.png"),
        show=show,
    )

    # Save animated GIF showing drones moving (3x speed for shorter file)
    animate_2d(
        primary, simulated, result,
        speed_factor=3.0,
        title="Scenario 1 – Conflict-Free Animation",
        save_path=os.path.join(output_dir, "scenario_clear_anim.gif"),
        show=show,
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    run(show=False)
