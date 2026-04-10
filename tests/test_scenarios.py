"""
Integration tests that run the full scenario pipelines and verify
end-to-end status outcomes without touching the filesystem.

Unlike unit tests (which test one function at a time), these tests
run the COMPLETE pipeline: build scenario → check conflicts → verify result.
No PNG or GIF files are created here — only the logic is tested.
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scenarios.scenario_clear    import build_scenario as build_clear
from scenarios.scenario_conflict import build_scenario as build_conflict
from scenarios.scenario_4d       import build_case_a, build_case_b
from src.deconfliction import check_conflicts


# ---------------------------------------------------------------------------
# Scenario 1 integration test — expects CLEAR
# ---------------------------------------------------------------------------

class TestScenarioClear(unittest.TestCase):

    def test_scenario_clear_returns_clear(self):
        # Run the full Scenario 1 pipeline and verify it returns 'clear'
        # SIM-1 departed too early, SIM-2 departs too late → no conflict
        primary, simulated = build_clear()
        result = check_conflicts(primary, simulated, safety_buffer=10.0)
        self.assertTrue(
            result.is_clear,
            f"Expected 'clear' but got conflicts: {result.conflicts}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 integration test — expects CONFLICT DETECTED
# ---------------------------------------------------------------------------

class TestScenarioConflict(unittest.TestCase):

    def test_scenario_conflict_returns_conflict_detected(self):
        # Run the full Scenario 2 pipeline — should find at least one conflict
        # SIM-A collides head-on, SIM-B near-miss, SIM-C tailgating
        primary, simulated = build_conflict()
        result = check_conflicts(primary, simulated, safety_buffer=8.0)
        self.assertFalse(
            result.is_clear,
            "Expected 'conflict detected' but got 'clear'"
        )

    def test_scenario_conflict_has_at_least_one_conflict(self):
        # The conflict list must not be empty — at least 1 conflict must be reported
        primary, simulated = build_conflict()
        result = check_conflicts(primary, simulated, safety_buffer=8.0)
        self.assertGreaterEqual(len(result.conflicts), 1)


# ---------------------------------------------------------------------------
# Scenario 3 integration tests — 3D space + time (extra credit)
# ---------------------------------------------------------------------------

class TestScenario4D(unittest.TestCase):

    def test_case_a_conflict(self):
        """3D crossing at same altitude and time → conflict."""
        # ALPHA-3D and SIM-X cross at (50,50,25) at t≈5.3s → 3D CONFLICT
        primary, simulated = build_case_a()
        result = check_conflicts(primary, simulated, safety_buffer=8.0)
        self.assertFalse(
            result.is_clear,
            "Expected 3D conflict in Case A"
        )

    def test_case_b_clear(self):
        """Same x-y crossing but altitude separation → clear."""
        # ALPHA-3D max altitude = 50m, SIM-HIGH flies at z=100 → 50m vertical gap → CLEAR
        # Proves system correctly checks 3D distance, not just 2D (x,y) distance
        primary, simulated = build_case_b()
        result = check_conflicts(primary, simulated, safety_buffer=8.0)
        self.assertTrue(
            result.is_clear,
            f"Expected 3D clear in Case B but got: {result.conflicts}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
