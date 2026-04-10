# Diagrams

Design and concept diagrams for the drone deconfliction system.

> **Note:** Planning notes and terminal commands during development were tracked in Notion.

## Files (5)

---

**`System_Architecture.jpg`**
High-level overview of the entire project. Shows how the main components connect:
`main.py` → `DeconflictionEngine` → `DroneMission` / `Waypoint` → `DeconflictionResult` → `html_report.py` → `report.html`

---

**`Data_Models_Diagram.png`**
Class diagram for the core data structures used in the system:
- `Waypoint` — a single point in 3D space with a timestamp
- `DroneMission` — list of waypoints + metadata (drone ID, speed, time window)
- `DeconflictionResult` — output of the engine (status: safe/conflict, list of conflict events, min distance)

---

**`Three_Scenarios_Overview.png`**
Visual summary of all test scenarios built into `main.py`:
- Scenario 1: 2D conflict — both drones at same altitude, paths cross within safety buffer
- Scenario 2: 2D clear — both drones at same altitude, paths never come close enough
- Scenario 3A: 3D conflict — both drones at z=0, conflict at intersection point
- Scenario 3B: 3D clear — primary drone climbs in altitude, separation keeps them safe

---

**`Conflict_Detection_Algorithm_Flowchart.jpg`**
Step-by-step flowchart of how `DeconflictionEngine.check()` works:
1. Validate input missions
2. Check feasibility (can the drone finish in its time window?)
3. For each segment pair — check temporal overlap
4. If overlap exists — compute closest approach using quadratic minimization
5. If minimum distance < safety buffer → record conflict
6. Return final result (safe or conflict with details)

---

**`line_diagrams.md`**
5 ASCII text diagrams that explain the math behind the algorithm — no images needed, readable in any text editor.

1. **Closest Approach Math** — how `dist²(t) = a·t² + b·t + c` is minimized to find the exact moment drones are closest
2. **`position_at_time()` Logic** — linear interpolation `pos = start + alpha × (end - start)` across multi-segment missions
3. **Temporal Overlap Pre-filter** — how the engine skips segment pairs that don't share any time (performance optimization)
4. **Safety Buffer Concept** — visual showing when 2 drones trigger CONFLICT (distance < 8m) vs SAFE (distance ≥ 8m)
5. **Feasibility Check** — how `mission_duration = total_distance / speed` is compared against the allowed time window
