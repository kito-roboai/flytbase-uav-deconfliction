"""
UAV Strategic Deconfliction System – Main Entry Point
======================================================
Runs all three scenarios and produces output files in the outputs/ directory.

Usage
-----
    python main.py                  # run all scenarios
    python main.py --scenario 1     # conflict-free only
    python main.py --scenario 2     # conflict scenarios only
    python main.py --scenario 3     # 4D (extra credit) only
    python main.py --show           # display plots interactively

Programmatic API example
------------------------
    from src import check_conflicts, DroneMission, Waypoint

    primary = DroneMission(
        drone_id="MY_DRONE",
        waypoints=[Waypoint(0, 0), Waypoint(100, 50), Waypoint(200, 0)],
        speed=15.0,
        start_time=0.0,
        mission_end=20.0,
    )
    simulated = [
        DroneMission("DRONE_B", [Waypoint(100,0), Waypoint(100,100)],
                     speed=10.0, start_time=3.0),
    ]
    result = check_conflicts(primary, simulated, safety_buffer=5.0)
    print(result.summary())
"""

from __future__ import annotations

import argparse
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def run_scenario_1(show: bool) -> None:
    from scenarios.scenario_clear import run
    print("\n" + "═" * 65)
    print("  SCENARIO 1 – Conflict-Free Mission")
    print("═" * 65)
    run(output_dir=OUTPUT_DIR, show=show)


def run_scenario_2(show: bool) -> None:
    from scenarios.scenario_conflict import run
    print("\n" + "═" * 65)
    print("  SCENARIO 2 – Conflict Detected (sample test case)")
    print("═" * 65)
    run(output_dir=OUTPUT_DIR, show=show)


def run_scenario_3(show: bool) -> None:
    from scenarios.scenario_4d import run
    print("\n" + "═" * 65)
    print("  SCENARIO 3 – 4D Deconfliction (3D Space + Time)  [Extra Credit]")
    print("═" * 65)
    run(output_dir=OUTPUT_DIR, show=show)


def generate_report() -> None:
    """Build interactive HTML report combining all three scenarios."""
    from scenarios.scenario_clear    import build_scenario as build1, SAFETY_BUFFER as BUF1
    from scenarios.scenario_conflict import build_scenario as build2, SAFETY_BUFFER as BUF2
    from scenarios.scenario_4d      import build_case_a, build_case_b, SAFETY_BUFFER as BUF3
    from src.deconfliction  import check_conflicts
    from src.html_report    import generate_html_report

    print("\n" + "═" * 65)
    print("  Generating Interactive HTML Report …")
    print("═" * 65)

    p1, s1 = build1()
    p2, s2 = build2()
    pa, sa = build_case_a()
    pb, sb = build_case_b()

    scenarios = [
        {
            "name":      "Scenario 1 — Conflict-Free Mission",
            "primary":   p1, "simulated": s1,
            "result":    check_conflicts(p1, s1, safety_buffer=BUF1),
            "buffer":    BUF1, "is_3d": False,
        },
        {
            "name":      "Scenario 2 — Conflict Detected",
            "primary":   p2, "simulated": s2,
            "result":    check_conflicts(p2, s2, safety_buffer=BUF2),
            "buffer":    BUF2, "is_3d": False,
        },
        {
            "name":      "Scenario 3A — 4D Conflict (3D space + time)",
            "primary":   pa, "simulated": sa,
            "result":    check_conflicts(pa, sa, safety_buffer=BUF3),
            "buffer":    BUF3, "is_3d": True,
        },
        {
            "name":      "Scenario 3B — 4D Clear (altitude separation)",
            "primary":   pb, "simulated": sb,
            "result":    check_conflicts(pb, sb, safety_buffer=BUF3),
            "buffer":    BUF3, "is_3d": True,
        },
    ]

    report_path = os.path.join(OUTPUT_DIR, "report.html")
    generate_html_report(scenarios, save_path=report_path)
    print(f"  Interactive report saved → {os.path.abspath(report_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UAV Strategic Deconfliction System"
    )
    parser.add_argument(
        "--scenario", type=int, choices=[1, 2, 3],
        help="Run only a specific scenario (1=clear, 2=conflict, 3=4D).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively instead of saving to disk.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.scenario == 1:
        run_scenario_1(args.show)
    elif args.scenario == 2:
        run_scenario_2(args.show)
    elif args.scenario == 3:
        run_scenario_3(args.show)
    else:
        run_scenario_1(args.show)
        run_scenario_2(args.show)
        run_scenario_3(args.show)
        generate_report()

    print("\n" + "═" * 65)
    print(f"  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("═" * 65)


if __name__ == "__main__":
    main()
