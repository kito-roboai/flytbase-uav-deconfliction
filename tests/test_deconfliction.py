"""
Unit tests for src/deconfliction.py

Each test encodes a hand-verifiable scenario so that expected results can be
derived analytically, not just empirically.
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DroneMission, Waypoint
from src.deconfliction import check_conflicts, DEFAULT_SAFETY_BUFFER


# ---------------------------------------------------------------------------
# Helper function — avoids repeating DroneMission(...) in every test
# ---------------------------------------------------------------------------

def straight_mission(
    drone_id: str,
    x0: float, y0: float,
    x1: float, y1: float,
    speed: float,
    start_time: float,
    mission_end: float = None,
    z0: float = 0.0,
    z1: float = 0.0,
) -> DroneMission:
    """Shortcut to create a single-segment (straight-line) mission."""
    return DroneMission(
        drone_id=drone_id,
        waypoints=[Waypoint(x0, y0, z0), Waypoint(x1, y1, z1)],
        speed=speed,
        start_time=start_time,
        mission_end=mission_end,
    )


# ---------------------------------------------------------------------------
# CLEAR scenarios — system should return status = 'clear'
# ---------------------------------------------------------------------------

class TestClearScenarios(unittest.TestCase):

    def test_no_simulated_drones(self):
        """No other drones → always clear."""
        # If there's nobody else in the sky, primary is always safe
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        result = check_conflicts(primary, [])
        self.assertTrue(result.is_clear)
        self.assertEqual(result.conflicts, [])  # empty conflict list

    def test_temporally_non_overlapping(self):
        """Simulated drone departs 1000 s later → no overlap in time."""
        # Paths cross at (50,50) but primary finishes at t=10, sim starts at t=1000
        # → they are never in the sky at the same time → CLEAR
        primary = straight_mission("P", 0, 50, 100, 50, speed=10, start_time=0)
        sim = straight_mission("S1", 50, 0, 50, 100, speed=10, start_time=1000)
        result = check_conflicts(primary, [sim], safety_buffer=10.0)
        self.assertTrue(result.is_clear)

    def test_large_spatial_separation(self):
        """Parallel paths 500 m apart → always clear."""
        # Both fly the same direction but 500m away — never get close
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 0, 500, 100, 500, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=10.0)
        self.assertTrue(result.is_clear)

    def test_perpendicular_crossing_with_time_miss(self):
        """
        Same intersection point but different velocities → drones are NOT
        there at the same time.

        Primary:   (0,50)→(100,50) speed=10 → reaches (50,50) at t=5 s
        Simulated: (50,0)→(50,100) speed=10, start_time=10 → reaches (50,50) at t=15 s
        Primary is done by t=10 → no overlap.
        """
        # Paths cross at (50,50) but primary passes at t=5, sim passes at t=15
        # → they miss each other in time → CLEAR
        primary = straight_mission("P", 0, 50, 100, 50, speed=10, start_time=0)
        sim = straight_mission("S1", 50, 0, 50, 100, speed=10, start_time=10)
        result = check_conflicts(primary, [sim], safety_buffer=8.0)
        self.assertTrue(result.is_clear)


# ---------------------------------------------------------------------------
# CONFLICT scenarios — system should return status = 'conflict detected'
# ---------------------------------------------------------------------------

class TestConflictScenarios(unittest.TestCase):

    def test_head_on_collision(self):
        """Two drones flying directly toward each other → definite conflict."""
        # P flies right, S1 flies left on the same line → they collide in the middle
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 100, 0, 0, 0, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)
        self.assertGreater(len(result.conflicts), 0)

    def test_perpendicular_crossing_at_same_time(self):
        """
        Primary:   (0,50)→(100,50) speed=10, t∈[0,10]  arrives at (50,50) at t=5
        Simulated: (50,0)→(50,100) speed=10, t∈[0,10]  arrives at (50,50) at t=5
        → distance = 0 at t=5 → definite conflict with any buffer > 0.
        """
        # Classic T-intersection: both arrive at (50,50) at exactly the same time
        primary = straight_mission("P", 0, 50, 100, 50, speed=10, start_time=0)
        sim = straight_mission("S1", 50, 0, 50, 100, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)
        self.assertEqual(result.conflicts[0].conflicting_drone_id, "S1")

    def test_conflict_time_is_accurate(self):
        """
        Head-on at speed 10 each, distance 100 m → meet at t=5 s at x=50.
        The reported conflict time should be close to 5 s.
        """
        # Verifies that the system reports the correct TIME of conflict
        # Two drones 100m apart, each at speed 10 → closing at 20 m/s → meet at t=5
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 100, 0, 0, 0, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)
        t = result.conflicts[0].time
        self.assertAlmostEqual(t, 5.0, delta=0.5)  # allow 0.5s tolerance

    def test_conflict_location_is_near_intersection(self):
        """Conflict location should be near the geometric crossing point."""
        # Verifies that the reported LOCATION of conflict is accurate
        # Crossing point is at (50,50) — conflict location should be within 5m of it
        import math
        primary = straight_mission("P", 0, 50, 100, 50, speed=10, start_time=0)
        sim = straight_mission("S1", 50, 0, 50, 100, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)
        x, y, z = result.conflicts[0].location
        dist_from_crossing = math.hypot(x - 50, y - 50)
        self.assertLess(dist_from_crossing, 5.0)

    def test_multiple_simulated_drones_multiple_conflicts(self):
        """Two simulated drones both conflict → two conflict records.

        Primary:   (0,50)→(100,50) speed=10, t∈[0,10]
          passes (50,50) at t=5 s and (30,50) at t=3 s

        S1: (50,0)→(50,100) speed=10, starts t=0 → at (50,50) at t=5 s → CONFLICT
        S2: (30,20)→(30,100) speed=10, starts t=0
            → travels 30 m to reach y=50 → arrives at (30,50) at t=3 s → CONFLICT
        """
        # Tests that the engine finds ALL conflicts, not just the first one
        primary = straight_mission("P", 0, 50, 100, 50, speed=10, start_time=0)
        sim1 = straight_mission("S1", 50, 0, 50, 100, speed=10, start_time=0)
        # S2 starts at y=20 → travels 30m to reach y=50 in 3s → meets primary at (30,50) at t=3
        sim2 = DroneMission(
            "S2",
            [Waypoint(30, 20), Waypoint(30, 100)],
            speed=10,
            start_time=0,
        )
        result = check_conflicts(primary, [sim1, sim2], safety_buffer=5.0)
        drone_ids = {c.conflicting_drone_id for c in result.conflicts}
        self.assertIn("S1", drone_ids)
        self.assertIn("S2", drone_ids)

    def test_same_lane_close_following(self):
        """Two drones on the same path separated by less than safety buffer."""
        # SIM starts 0.3s late → gap = speed × delay = 10 × 0.3 = 3m < 5m buffer → CONFLICT
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 0, 0, 100, 0, speed=10, start_time=0.3)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)

    def test_reported_separation_less_than_buffer(self):
        """Conflict separation must always be less than safety buffer."""
        # Any reported conflict must have separation < buffer — basic sanity check
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 100, 0, 0, 0, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        for c in result.conflicts:
            self.assertLess(c.separation, c.safety_buffer)


# ---------------------------------------------------------------------------
# Feasibility tests — checks if mission can finish within the time window
# ---------------------------------------------------------------------------

class TestFeasibility(unittest.TestCase):

    def test_feasible_within_window(self):
        """Mission completes well within the window → feasible=True."""
        # Mission takes 10s, window is 20s → plenty of time → feasible
        primary = straight_mission(
            "P", 0, 0, 100, 0, speed=10, start_time=0, mission_end=20.0
        )
        result = check_conflicts(primary, [])
        self.assertTrue(result.feasible)

    def test_infeasible_too_slow(self):
        """Mission requires 10 s but window is only 5 s → feasible=False."""
        # Mission takes 10s but deadline is at t=5 → drone can't finish in time
        primary = straight_mission(
            "P", 0, 0, 100, 0, speed=10, start_time=0, mission_end=5.0
        )
        result = check_conflicts(primary, [])
        self.assertFalse(result.feasible)

    def test_no_mission_end_always_feasible(self):
        """When mission_end is None, feasibility is not constrained."""
        # No deadline set → feasibility check is skipped → always True
        primary = straight_mission("P", 0, 0, 100, 0, speed=1, start_time=0)
        result = check_conflicts(primary, [])
        self.assertTrue(result.feasible)


# ---------------------------------------------------------------------------
# 3D conflict tests — altitude (z) matters in conflict detection
# ---------------------------------------------------------------------------

class Test3DConflicts(unittest.TestCase):

    def test_3d_same_altitude_conflict(self):
        """Drones crossing at the same altitude → conflict."""
        # Both fly at z=25 and their (x,y) paths cross → 3D conflict
        primary = DroneMission(
            "P3D",
            [Waypoint(0, 50, 25), Waypoint(100, 50, 25)],
            speed=10, start_time=0,
        )
        sim = DroneMission(
            "S3D",
            [Waypoint(50, 0, 25), Waypoint(50, 100, 25)],
            speed=10, start_time=0,
        )
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result.is_clear)

    def test_3d_altitude_separation_clear(self):
        """Same x-y crossing but 50 m altitude difference → clear."""
        # Primary flies at z=0, SIM flies at z=50 → 50m vertical gap → CLEAR
        # This proves the system uses 3D distance, not just 2D
        primary = DroneMission(
            "P3D",
            [Waypoint(0, 50, 0), Waypoint(100, 50, 0)],
            speed=10, start_time=0,
        )
        sim = DroneMission(
            "S3D",
            [Waypoint(50, 0, 50), Waypoint(50, 100, 50)],
            speed=10, start_time=0,
        )
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertTrue(result.is_clear)


# ---------------------------------------------------------------------------
# Edge case tests — unusual inputs that should not crash the system
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_very_small_safety_buffer(self):
        """With near-zero buffer, only an exact collision triggers."""
        # Drones are 1m apart (parallel lanes) — with buffer=0.5m they are safe
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 0, 1, 100, 1, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=0.5)
        self.assertTrue(result.is_clear)

    def test_large_number_of_simulated_drones(self):
        """Engine must handle many drones without error."""
        # 20 drones all flying parallel at different y offsets — none conflicts
        # Tests that the engine scales without crashing or slowing down badly
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sims = [
            straight_mission(f"S{i}", 0, float(i * 50), 100, float(i * 50),
                             speed=10, start_time=0)
            for i in range(1, 21)   # 20 simulated drones, each 50m apart
        ]
        result = check_conflicts(primary, sims, safety_buffer=5.0)
        self.assertTrue(result.is_clear)

    def test_degenerate_zero_length_segment_in_simulated(self):
        """A repeated waypoint (zero-length segment) should not crash."""
        # Waypoint(50,50) appears twice → segment length = 0 → potential divide-by-zero
        # System should handle this gracefully without crashing
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = DroneMission(
            "HOVER",
            [Waypoint(50, 50), Waypoint(50, 50), Waypoint(60, 60)],  # duplicate waypoint
            speed=5, start_time=0,
        )
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        # Just needs to run without crashing — either result is acceptable
        self.assertIn(result.status, {"clear", "conflict detected"})

    def test_result_summary_is_string(self):
        # summary() must return a string (for printing to terminal)
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 100, 0, 0, 0, speed=10, start_time=0)
        result = check_conflicts(primary, [sim], safety_buffer=5.0)
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Status", summary)  # summary must contain the word "Status"

    def test_custom_safety_buffer_respected(self):
        """With a huge safety buffer, parallel drones 50 m apart conflict."""
        # Same two drones, same paths — but different buffer sizes
        # buffer=50: drones 30m apart → inside buffer → CONFLICT
        # buffer=5:  drones 30m apart → outside buffer → CLEAR
        primary = straight_mission("P", 0, 0, 100, 0, speed=10, start_time=0)
        sim = straight_mission("S1", 0, 30, 100, 30, speed=10, start_time=0)
        result_large = check_conflicts(primary, [sim], safety_buffer=50.0)
        result_small = check_conflicts(primary, [sim], safety_buffer=5.0)
        self.assertFalse(result_large.is_clear)  # 30m < 50m buffer → conflict
        self.assertTrue(result_small.is_clear)   # 30m > 5m buffer → clear


if __name__ == "__main__":
    unittest.main(verbosity=2)
