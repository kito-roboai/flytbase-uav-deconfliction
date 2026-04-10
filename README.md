# UAV Strategic Deconfliction in Shared Airspace

**Version:** 1.1 | **Language:** Python 3.10 | **OS:** Ubuntu 22.04 LTS

A strategic deconfliction system that serves as the final authority for verifying whether a drone's planned waypoint mission is safe to execute in shared airspace. Before a drone takes off, this system checks its planned path against all other drones in the airspace and returns **CLEAR** or **CONFLICT DETECTED** — with exact time, location, and separation for every violation.

---

## Features

- **Continuous trajectory analysis** – positions computed as closed-form analytic functions of time; no discrete time-stepping (zero missed conflicts).
- **Exact conflict detection** – inter-drone distance minimised as a quadratic function of time for each segment pair, O(1) per pair.
- **Rich conflict reports** – time, location (x, y, z), and separation for every violation.
- **Feasibility check** – verifies the mission can complete within the given time window.
- **2D & 3D static plots**, **animated GIFs**, and **4D animation** (3D space + time).
- **Interactive HTML report** – single-file report with live canvas animation, drag-to-rotate 3D view, play/pause/speed controls; open in any browser, no internet needed.
- **Comprehensive test suite** – 46 unit + integration tests, all passing in 0.014 seconds.

---

## Technologies & Tools

### Libraries & Versions

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Core language |
| numpy | 2.2.6 | Vector math — positions, distances, dot products |
| matplotlib | 3.10.8 | 2D/3D static plots and GIF animations |
| Pillow | 12.2.0 | GIF frame saving (used by matplotlib) |
| plotly | 6.7.0 | Interactive 3D path plots in HTML report |
| pytest | 9.0.3 | Test runner |

> Install all with: `pip install -r requirements.txt`

### AI Tools Used

| Tool | How It Was Used |
|------|----------------|
| **Claude Code** (Anthropic) | Main development partner — architecture, math logic, code writing, visualization boilerplate, HTML report engine |
| **Amazon Q** | Code review, understanding generated code, documentation help |
| **ChatGPT** | Concept clarification, math explanations |
| **Gemini** | Cross-checking ideas, alternative approaches |

### Design & Diagram Tools

| Tool | Purpose |
|------|---------|
| **Draw.io** | System architecture diagram, algorithm flowchart, data model diagram |
| **VS Code** | Code editor with Claude Code extension |
| **Notion** | Planning notes, task tracking, and terminal command reference during development |

---

## Project Structure

```
flytbase_assignment/
├── src/
│   ├── __init__.py           # Public API exports
│   ├── models.py             # Waypoint, DroneMission, ConflictDetail, DeconflictionResult
│   ├── trajectory.py         # Analytic position computation, segment-pair closest approach
│   ├── deconfliction.py      # Core conflict detection engine
│   ├── visualization.py      # 2D/3D/4D plotting and GIF animation
│   ├── html_report.py        # Interactive HTML report with live canvas animation
│   └── utils.py              # Shared utility helpers
├── scenarios/
│   ├── scenario_clear.py     # Scenario 1: conflict-free mission
│   ├── scenario_conflict.py  # Scenario 2: conflict detected (spec sample test case)
│   └── scenario_4d.py        # Scenario 3: 4D extra credit (3D space + time)
├── tests/
│   ├── test_trajectory.py    # Unit tests for trajectory math
│   ├── test_deconfliction.py # Unit tests for conflict engine
│   └── test_scenarios.py     # Integration tests (end-to-end scenario checks)
├── Diagrams/                 # System architecture, flowchart, data model diagrams
├── outputs/                  # Generated PNG plots, GIF animations, HTML report
├── main.py                   # CLI entry point — runs all scenarios
├── requirements.txt
└── REFLECTION.md             # System design & reflection document
```

---

## Quick Start

```bash
# Step 1 — Activate virtual environment (run from project root)
cd /path/to/flytbase_assignment
source venv/bin/activate

# Step 2 — Run all 3 scenarios
python main.py --scenario 1
python main.py --scenario 2
python main.py --scenario 3

# Step 3 — Run tests
python -m unittest discover -s tests -v

# Step 4 — Check output files
ls outputs/
```

Then open `outputs/report.html` in your browser for the full interactive report.

---

## Setup (First Time)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the System

### Run all scenarios
```bash
python main.py
```

### Run a specific scenario
```bash
python main.py --scenario 1   # conflict-free (2D)
python main.py --scenario 2   # conflict detected (2D)
python main.py --scenario 3   # 4D extra credit (3D space + time)
```

All outputs are saved to the `outputs/` directory.  
**Primary output:** open `outputs/report.html` in any browser for the full interactive report.

---

## Running the Tests

```bash
# Recommended (avoids ROS2 plugin conflicts on Ubuntu with ROS installed)
python -m unittest discover -s tests -v

# Or with pytest
python -m pytest tests/ -v
```

**46 tests, all passing. Runtime: ~0.014 seconds.**

---

## How It Works (Brief)

```
INPUT
  ├── Primary drone: waypoints (x,y,z), speed, start_time, deadline
  └── Simulated drones: same fields for each

PROCESS
  1. Feasibility check — can the drone finish within its time window?
  2. For every segment pair (primary segment × simulated segment):
       a. Find time window when BOTH drones are on their segment (overlap)
       b. Express relative displacement: D(t) = P(t) - Q(t)  [linear in t]
       c. |D(t)|² = at² + bt + c  [quadratic = parabola]
       d. Minimum at vertex: t_min = -b / (2a)  [O(1), no loop]
       e. If min distance < safety_buffer → CONFLICT

OUTPUT
  ├── Status: CLEAR or CONFLICT DETECTED
  ├── Feasible: True / False
  ├── Mission duration: X seconds
  └── Conflict details: drone ID, time, location (x,y,z), separation
```

---

## Programmatic API

```python
from src import check_conflicts, DroneMission, Waypoint

primary = DroneMission(
    drone_id="MY_DRONE",
    waypoints=[Waypoint(0, 0), Waypoint(100, 50), Waypoint(200, 0)],
    speed = 15.0,
    start_time = 0.0,
    mission_end = 20.0,
)

simulated = [
    DroneMission(
        drone_id = "DRONE_B",
        waypoints = [Waypoint(100, 0), Waypoint(100, 100)],
        speed = 10.0,
        start_time = 3.0,
    ),
]

result = check_conflicts(primary, simulated, safety_buffer = 5.0)
print(result.summary())
# Status         : CLEAR
# Feasible       : True
# Mission duration: 15.81 s
```

---

## Output Files

| File | Description |
|------|-------------|
| `report.html` | **Primary output** — interactive report with live 2D/3D canvas animation, all scenarios, open in browser |
| `scenario_clear_2d.png` | Trajectory overview – conflict-free |
| `scenario_clear_dist.png` | Separation vs time – conflict-free |
| `scenario_clear_anim.gif` | Animated drone flight – conflict-free |
| `scenario_conflict_2d.png` | Trajectory overview with red conflict markers |
| `scenario_conflict_dist.png` | Separation vs time showing violation |
| `scenario_conflict_anim.gif` | Animated drone flight – conflict scenario |
| `scenario_4d_case_a_3d.png` | 3D isometric plot – altitude conflict |
| `scenario_4d_case_a_anim.gif` | 4D animation – altitude conflict |
| `scenario_4d_case_b_3d.png` | 3D isometric plot – altitude separation (clear) |
| `scenario_4d_case_b_anim.gif` | 4D animation – altitude separation (clear) |
