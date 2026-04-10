"""
Unit tests for src/trajectory.py

Tests verify that position is computed as a *continuous analytic function*
of time, not via discrete stepping.
"""

import math
import unittest

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DroneMission, Waypoint
from src.trajectory import (
    compute_segment_times,
    mission_duration,
    mission_end_time,
    position_at_time,
    closest_approach_on_segment_pair,
    sample_trajectory,
)


def make_simple_mission(speed=10.0, start_time=0.0):
    """
    Helper: creates a simple 3-segment mission for reuse across tests.
    Path: (0,0) → (10,0) → (10,10) → (20,10)
    Each segment is 10m long → at speed=10, each takes 1 second.
    """
    return DroneMission(
        drone_id="TEST",
        waypoints=[
            Waypoint(0, 0),
            Waypoint(10, 0),
            Waypoint(10, 10),
            Waypoint(20, 10),
        ],
        speed=speed,
        start_time=start_time,
    )


# ---------------------------------------------------------------------------
# Tests for compute_segment_times()
# ---------------------------------------------------------------------------

class TestSegmentTimes(unittest.TestCase):

    def test_segment_count(self):
        # 4 waypoints → 3 segments (A→B, B→C, C→D)
        m = make_simple_mission()
        segs = compute_segment_times(m)
        self.assertEqual(len(segs), 3)

    def test_first_segment_start_equals_mission_start(self):
        # If drone departs at t=5, first segment must start at t=5 (not t=0)
        m = make_simple_mission(start_time=5.0)
        segs = compute_segment_times(m)
        self.assertAlmostEqual(segs[0][0], 5.0)

    def test_segment_duration_matches_distance_over_speed(self):
        """Each segment time = segment_length / speed."""
        m = make_simple_mission(speed=10.0)
        segs = compute_segment_times(m)
        # Each segment is 10m long, speed=10 m/s → each takes exactly 1.0 second
        for t_start, t_end in segs:
            self.assertAlmostEqual(t_end - t_start, 1.0)

    def test_segments_are_contiguous(self):
        """End of segment i == start of segment i+1."""
        # No gaps allowed — drone must be somewhere at every moment during flight
        m = make_simple_mission()
        segs = compute_segment_times(m)
        for i in range(len(segs) - 1):
            self.assertAlmostEqual(segs[i][1], segs[i + 1][0])

    def test_total_duration(self):
        # Total path = 30m (10+10+10), speed=5 → takes 6 seconds
        m = make_simple_mission(speed=5.0)
        self.assertAlmostEqual(mission_duration(m), 6.0)


# ---------------------------------------------------------------------------
# Tests for position_at_time()
# ---------------------------------------------------------------------------

class TestPositionAtTime(unittest.TestCase):

    def setUp(self):
        # Simple horizontal mission: (0,0) → (100,0), speed=10, takes 10 seconds
        self.m = DroneMission(
            drone_id="P",
            waypoints=[Waypoint(0, 0), Waypoint(100, 0)],
            speed=10.0,
            start_time=0.0,
        )

    def test_position_at_start(self):
        # At t=0, drone should be exactly at first waypoint (0,0,0)
        pos = position_at_time(self.m, 0.0)
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-9)

    def test_position_at_end(self):
        # At t=10, drone should be exactly at last waypoint (100,0,0)
        pos = position_at_time(self.m, 10.0)
        np.testing.assert_allclose(pos, [100, 0, 0], atol=1e-9)

    def test_position_at_midpoint(self):
        # At t=5 (halfway through 10s mission), drone should be at x=50
        pos = position_at_time(self.m, 5.0)
        np.testing.assert_allclose(pos, [50, 0, 0], atol=1e-9)

    def test_position_outside_window_returns_none(self):
        # Before mission starts or after it ends → drone has no position → None
        self.assertIsNone(position_at_time(self.m, -1.0))   # before departure
        self.assertIsNone(position_at_time(self.m, 11.0))   # after landing

    def test_position_is_continuous(self):
        """Sample many times and verify no jumps > speed * dt."""
        # Checks that drone never "teleports" — movement should be smooth
        times = np.linspace(0, 10, 1000)
        positions = [position_at_time(self.m, t) for t in times]
        dt = times[1] - times[0]
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            # At speed=10, in time dt, max movement = speed * dt
            self.assertLessEqual(dist, self.m.speed * dt + 1e-9)

    def test_3d_position(self):
        """Verify z interpolation works correctly."""
        # Drone flies straight up: (0,0,0) → (0,0,100), speed=10 → takes 10s
        # At t=5 it should be at altitude z=50
        m3d = DroneMission(
            drone_id="3D",
            waypoints=[Waypoint(0, 0, 0), Waypoint(0, 0, 100)],
            speed=10.0,
            start_time=0.0,
        )
        pos = position_at_time(m3d, 5.0)
        np.testing.assert_allclose(pos, [0, 0, 50], atol=1e-9)

    def test_nonzero_start_time(self):
        # Drone departs at t=10 → asking position at t=5 should return None
        # Asking at t=15 (5s after departure) → should be at x=50
        m = DroneMission(
            drone_id="T",
            waypoints=[Waypoint(0, 0), Waypoint(100, 0)],
            speed=10.0,
            start_time=10.0,
        )
        self.assertIsNone(position_at_time(m, 5.0))    # before departure → None
        pos = position_at_time(m, 15.0)                # 5s after departure → x=50
        np.testing.assert_allclose(pos, [50, 0, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# Tests for closest_approach_on_segment_pair()
# ---------------------------------------------------------------------------

class TestClosestApproach(unittest.TestCase):

    def test_head_on_collision(self):
        """Two drones flying toward each other at the same speed meet in the middle."""
        # Drone 1: (0,0) → (100,0)  moving right
        # Drone 2: (100,0) → (0,0)  moving left
        # Both at speed 10 m/s → meet at x=50 at t=5s → distance = 0
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([100.0, 0.0, 0.0])
        q1 = np.array([100.0, 0.0, 0.0])
        q2 = np.array([0.0, 0.0, 0.0])
        t_close, pos_p, pos_q, dist = closest_approach_on_segment_pair(
            p1, p2, 0.0, 10.0,
            q1, q2, 0.0, 10.0,
        )
        self.assertAlmostEqual(dist, 0.0, places=6)     # they actually touch
        self.assertAlmostEqual(t_close, 5.0, places=6)  # at t=5 seconds

    def test_parallel_same_speed(self):
        """Parallel drones at fixed lateral offset stay at constant distance."""
        # Drone 1 flies along y=0, Drone 2 flies along y=20 (always 20m apart)
        # Since they move at the same speed in the same direction, gap never changes
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([100.0, 0.0, 0.0])
        q1 = np.array([0.0, 20.0, 0.0])
        q2 = np.array([100.0, 20.0, 0.0])
        _, _, _, dist = closest_approach_on_segment_pair(
            p1, p2, 0.0, 10.0,
            q1, q2, 0.0, 10.0,
        )
        self.assertAlmostEqual(dist, 20.0, places=6)  # always 20m apart

    def test_no_temporal_overlap_returns_inf(self):
        """Non-overlapping time intervals → distance = inf."""
        # Drone 1 flies t=0 to t=5, Drone 2 flies t=10 to t=15
        # They never share airspace at the same time → no conflict possible
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([10.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0])
        q2 = np.array([10.0, 0.0, 0.0])
        _, _, _, dist = closest_approach_on_segment_pair(
            p1, p2, 0.0, 5.0,
            q1, q2, 10.0, 15.0,  # completely different time window
        )
        self.assertEqual(dist, float("inf"))  # infinity = no overlap = safe

    def test_perpendicular_crossing_at_known_time(self):
        """
        Primary:   (0,50) → (100,50) at speed 10, t∈[0,10]  → at x=50, t=5
        Simulated: (50,0) → (50,100) at speed 10, t∈[0,10]  → at y=50, t=5
        Distance at t=5 should be 0.
        """
        # Classic T-intersection collision — both reach (50,50) at exactly t=5
        p1 = np.array([0.0, 50.0, 0.0])
        p2 = np.array([100.0, 50.0, 0.0])
        q1 = np.array([50.0, 0.0, 0.0])
        q2 = np.array([50.0, 100.0, 0.0])
        t_close, pos_p, pos_q, dist = closest_approach_on_segment_pair(
            p1, p2, 0.0, 10.0,
            q1, q2, 0.0, 10.0,
        )
        self.assertAlmostEqual(dist, 0.0, places=5)    # distance = 0 (exact collision)
        self.assertAlmostEqual(t_close, 5.0, places=5) # at t=5 seconds

    def test_sample_trajectory_shape(self):
        # sample_trajectory returns (times array, positions array)
        # with exactly n_points rows
        m = make_simple_mission()
        times, positions = sample_trajectory(m, n_points=50)
        self.assertEqual(times.shape, (50,))       # 50 time values
        self.assertEqual(positions.shape, (50, 3)) # 50 positions, each [x,y,z]


# ---------------------------------------------------------------------------
# Tests for input validation in DroneMission
# ---------------------------------------------------------------------------

class TestInputValidation(unittest.TestCase):

    def test_single_waypoint_raises(self):
        # A mission needs at least 2 waypoints to define a path — 1 is not enough
        with self.assertRaises(ValueError):
            DroneMission("X", [Waypoint(0, 0)], speed=5.0, start_time=0.0)

    def test_zero_speed_raises(self):
        # Speed of 0 means drone never moves — not a valid mission
        with self.assertRaises(ValueError):
            DroneMission("X", [Waypoint(0, 0), Waypoint(1, 0)], speed=0.0, start_time=0.0)

    def test_negative_speed_raises(self):
        # Negative speed makes no physical sense — must be rejected
        with self.assertRaises(ValueError):
            DroneMission("X", [Waypoint(0, 0), Waypoint(1, 0)], speed=-5.0, start_time=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
