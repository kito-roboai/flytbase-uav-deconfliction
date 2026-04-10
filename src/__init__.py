"""
UAV Strategic Deconfliction System
====================================
Modules:
    models        - Data structures for drones, waypoints, and results
    trajectory    - Continuous trajectory math (position as a function of time)
    deconfliction - Core conflict detection engine
    visualization - 2D / 3D / 4D plotting and animation
"""
from .models import Waypoint, DroneMission, ConflictDetail, DeconflictionResult
from .deconfliction import check_conflicts

__all__ = [
    "Waypoint",
    "DroneMission",
    "ConflictDetail",
    "DeconflictionResult",
    "check_conflicts",
]
