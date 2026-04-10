"""
Data models for the UAV Strategic Deconfliction System.

All positions are in a consistent unit (e.g. metres).
All times are in seconds from an arbitrary epoch (t=0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """A 3-D spatial position (z defaults to 0 for 2-D missions)."""
    x: float   # horizontal position in metres (left-right)
    y: float   # horizontal position in metres (forward-back)
    z: float = 0.0  # altitude in metres (default = 0, i.e., ground level)

    def to_array(self) -> np.ndarray:
        """Return waypoint as a NumPy array [x, y, z].
        Used in math calculations (trajectory, distance)."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def __repr__(self) -> str:
        return f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


# ---------------------------------------------------------------------------
# Mission definitions
# ---------------------------------------------------------------------------

@dataclass
class DroneMission:
    """
    A complete drone mission specification.

    Attributes
    ----------
    drone_id      : Unique identifier string.
    waypoints     : Ordered list of waypoints defining the route.
    speed         : Constant travel speed (units / second).
    start_time    : Departure time from the first waypoint (seconds).
    mission_end   : Latest time by which the primary mission must finish.
                    Only meaningful for the primary drone; ignored for simulated
                    drones (set to None).
    """
    drone_id: str                    # name of the drone e.g. "ALPHA", "SIM-1"
    waypoints: List[Waypoint]        # list of points the drone will fly through (in order)
    speed: float                     # constant speed in m/s (no acceleration modelled)
    start_time: float                # time (seconds) when drone leaves first waypoint
    mission_end: Optional[float] = None  # deadline — only set for primary drone

    def __post_init__(self) -> None:
        # Validate: need at least 2 points to define a path
        if len(self.waypoints) < 2:
            raise ValueError(
                f"Mission '{self.drone_id}' must have at least 2 waypoints."
            )
        # Validate: speed must be positive (no hovering or reverse)
        if self.speed <= 0:
            raise ValueError(
                f"Mission '{self.drone_id}' speed must be positive, got {self.speed}."
            )

    @property
    def is_3d(self) -> bool:
        """True if any waypoint has a non-zero z coordinate.
        Used to decide whether to show 2D or 3D plots."""
        return any(wp.z != 0.0 for wp in self.waypoints)


# ---------------------------------------------------------------------------
# Conflict / result types
# ---------------------------------------------------------------------------

@dataclass
class ConflictDetail:
    """
    A single spatio-temporal conflict between the primary drone and one
    simulated drone.

    Attributes
    ----------
    conflicting_drone_id : ID of the simulated drone that caused the conflict.
    time                 : Time (seconds) at which the closest approach occurs.
    location             : (x, y, z) coordinates of the closest approach point
                           on the primary drone's trajectory.
    separation           : Actual minimum separation distance at conflict time.
    safety_buffer        : The required minimum separation that was violated.
    """
    conflicting_drone_id: str              # which simulated drone caused this conflict
    time: float                            # at what time (seconds) the conflict happens
    location: Tuple[float, float, float]   # (x, y, z) position of primary drone at conflict
    separation: float                      # actual distance between the two drones (< safety_buffer)
    safety_buffer: float                   # minimum allowed distance that was violated

    def __repr__(self) -> str:
        x, y, z = self.location
        return (
            f"ConflictDetail("
            f"drone='{self.conflicting_drone_id}', "
            f"t={self.time:.2f}s, "
            f"loc=({x:.2f}, {y:.2f}, {z:.2f}), "
            f"sep={self.separation:.2f} < buffer={self.safety_buffer:.2f})"
        )


@dataclass
class DeconflictionResult:
    """
    Output of the deconfliction query.

    Attributes
    ----------
    status           : 'clear' or 'conflict detected'.
    conflicts        : List of ConflictDetail objects (empty when status='clear').
    feasible         : Whether the primary mission can complete within its time
                       window (always True when mission_end is None).
    actual_duration  : Calculated time required to complete the primary mission.
    """
    status: str                                        # "clear" or "conflict detected"
    conflicts: List[ConflictDetail] = field(default_factory=list)  # all conflict events found
    feasible: bool = True                              # can mission finish before deadline?
    actual_duration: float = 0.0                       # how long the mission actually takes (seconds)

    @property
    def is_clear(self) -> bool:
        # Returns True only when no conflicts were found
        return self.status == "clear"

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Status         : {self.status.upper()}",
            f"Feasible       : {self.feasible}",
            f"Mission duration: {self.actual_duration:.2f} s",
        ]
        if self.conflicts:
            lines.append(f"Conflicts found: {len(self.conflicts)}")
            for i, c in enumerate(self.conflicts, 1):
                x, y, z = c.location
                lines.append(
                    f"  [{i}] Drone '{c.conflicting_drone_id}' | "
                    f"t={c.time:.2f}s | "
                    f"loc=({x:.2f},{y:.2f},{z:.2f}) | "
                    f"sep={c.separation:.2f}m (buffer={c.safety_buffer:.2f}m)"
                )
        return "\n".join(lines)
