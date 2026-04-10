# Reflection & Justification Document

**Project:** UAV Strategic Deconfliction in Shared Airspace (v1.1)
**Author:** Krishna Teja | M.Tech Robotics & AI

---

## 1. Design Decisions & Architectural Choices

The system is split into five focused modules that mirror the data pipeline:

| Module | Responsibility |
|--------|---------------|
| `models.py` | Immutable data contracts (Waypoint, DroneMission, results) |
| `trajectory.py` | Analytic geometry — position as a closed-form function of time |
| `deconfliction.py` | Conflict detection engine — calls trajectory math, aggregates results |
| `visualization.py` | All rendering concerns, fully decoupled from detection logic |
| `html_report.py` | Interactive HTML report — live canvas animation, 3D isometric view |

This layering means each module can be tested, replaced, or scaled independently. For instance, swapping `deconfliction.py` for a distributed version requires zero changes to the other four modules.

**Rule enforced:** Lower modules never import from upper modules.
- `models.py` → knows nothing
- `trajectory.py` → knows only `models.py`
- `deconfliction.py` → knows `trajectory.py` and `models.py`
- `visualization.py` → knows `trajectory.py` and `models.py`
- `main.py` → knows everything

### Why analytic (not discrete) trajectory computation?

The spec explicitly forbids "solutions that rely on pre-calculated positions at discrete time steps." Beyond spec compliance, discrete sampling introduces false negatives: two fast drones can pass through each other between frames if the time step is too coarse. The analytic approach eliminates this entirely.

**Implementation:** For each pair of flight segments, positions are expressed as:

```
P(t) = p_start + v_P * (t − t_P_start)
Q(t) = q_start + v_Q * (t − t_Q_start)
```

The relative displacement `D(t) = P(t) − Q(t)` is linear in `t`, so `|D(t)|²` is a quadratic polynomial. Finding its minimum over the temporal overlap interval takes O(1) — just evaluating the vertex of the parabola (`t = -b / (2a)`) and the two endpoints. This gives the exact closest-approach time, location, and distance with zero sampling error.

---

## 2. Spatial and Temporal Checks

**Spatial check** is embedded inside the analytic closest-approach computation. If the minimum separation over a time interval falls below the safety buffer, a spatial violation is recorded.

**Temporal check** is enforced by the overlapping interval `[max(t1_start, t2_start), min(t1_end, t2_end)]`. If this interval is empty, the segment pair is skipped — the drones simply are not there at the same time, regardless of spatial proximity.

**Feasibility check** compares the calculated mission completion time against `mission_end`. A mission can be spatially clear yet infeasible if the drone cannot cover all waypoints at its given speed within the window.

```
mission_duration = total_distance / speed
If mission_duration > (mission_end - start_time) → INFEASIBLE
```

---

## 3. AI Tools & Integration

Four AI tools were used at different stages of development:

| Tool | Role |
|------|------|
| **Claude Code** (Anthropic) | Primary development partner throughout — architecture discussion, implementing math logic, writing test scaffolding, building visualization boilerplate, creating the full interactive HTML report engine with 2D/3D canvas animation |
| **Amazon Q** | Code review — reading and explaining generated code, identifying edge cases, documentation structure |
| **ChatGPT** | Concept clarification — explaining quadratic minimization, parabola vertex formula, and linear interpolation intuitively |
| **Gemini** | Cross-checking architectural decisions and alternative approaches |

### What I (human) did vs what AI did

```
HUMAN                                       AI
────────────────────────────────────────────────────────
Decided the architecture                    Wrote boilerplate for each module
(4 modules: data/math/logic/visual)

Chose analytic over discrete sampling       Expressed the math as clean Python

Designed the parabola minimization          Generated initial test scaffolding
approach (D(t) = A + Bt, |D|² = quadratic)

Verified all math by hand calculation:      Wrote the FuncAnimation setup for GIF,
ALPHA at x=50 at t=5s ✓                    matplotlib 3D subplot config, and
SIM-A at y=50 at t=5s ✓                    HTML canvas animation engine (~300 lines)
separation = 0 ✓

Caught and fixed geometry bugs in           Fixed tests after I corrected the geometry
2 test cases (SIM-2 position error)

Made all design decisions                   Accelerated the implementation speed
```

**Critical evaluation principle:** All AI-generated code was verified against hand-calculated examples before acceptance. If generated code disagreed with manual math, the code was wrong — not the math.

---

## 4. Interactive HTML Report (Extra Feature)

Beyond static plots and GIFs, `html_report.py` generates a fully self-contained `report.html` — no server, no internet, opens in any browser.

**Features:**
- Live canvas animation for each scenario (play/pause, speed control, time scrubber)
- 2D scenarios: top-down animated flight with heading arrows, trails, pulsing conflict markers
- 3D scenarios: isometric oblique projection with drag-to-rotate, altitude shadows (dashed line from drone to ground), Z-axis ruler with altitude ticks
- 4D nature: 3D space (x, y, z) animated over time — the "4th dimension"
- All data embedded as JSON in the HTML — fully portable

This was implemented as a separate IIFE-isolated JavaScript module (`_CANVAS_3D_JS`) that shares no scope with the 2D engine and dispatches based on `data.is3d` flag.

---

## 5. Testing Strategy & Edge Cases

### Strategy

- **Unit tests for math** (`test_trajectory.py`): Verify each mathematical primitive in isolation — segment timing, position interpolation, closest-approach minimisation.
- **Unit tests for logic** (`test_deconfliction.py`): Cover every logical branch — no drones, temporal non-overlap, spatial separation, head-on, perpendicular, same-lane, 3D altitude separation.
- **Integration tests** (`test_scenarios.py`): Run the full end-to-end pipeline for each scenario and assert the correct overall status.

**Total: 46 tests, all passing. Runtime: ~0.014 seconds.**

### Key edge cases handled

| Edge Case | Handling |
|-----------|----------|
| Zero-length segment (repeated waypoint) | `dt=0` check → treated as hover point |
| Non-overlapping time windows | Early exit: `t_lo >= t_hi` → distance = ∞ |
| Primary mission outside time window | Segments clamped to `[mission_start, mission_end]` |
| Parallel identical paths | Relative velocity = 0 → distance is constant (checked at endpoints) |
| Very small safety buffer | Works correctly; buffer=0 only triggers exact zero-distance events |
| Many simulated drones (20+) | O(M×N) but no architectural limit; tested in `test_large_number_of_simulated_drones` |

---

## 6. Scaling to Tens of Thousands of Drones

The current system is single-process and O(M × N) per query, which is fine for tens of drones but cannot handle a real-world UTM deployment. Here is the path to production scale:

### 6.1 Spatial Indexing
Replace the naive all-pairs check with a **3D R-tree or k-d tree** on bounding boxes of trajectory segments. For a query drone with M segments, this reduces the candidate set from O(N×K) to O(M × log(N×K)) where K is average segments per drone.

### 6.2 Temporal Indexing
Maintain a **time-bucketed index** (e.g. per-minute slots). A query only loads drones whose flight windows overlap the primary mission's window. This is the dominant win for sparse, real-world traffic.

### 6.3 Distributed Architecture
```
                  ┌──────────────────────────┐
 Drone Client ──► │  API Gateway (load balancer) │
                  └──────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Worker 1       Worker 2       Worker N
         (region A)     (region B)     (region C)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    Shared Flight Plan Store
                    (e.g. Redis + PostGIS)
```

- Partition airspace into geographic cells; each worker owns a cell and a buffer zone.
- Queries crossing cell boundaries are replicated to adjacent workers.
- Workers publish conflict notifications to a message queue (Kafka / NATS).

### 6.4 Real-Time Ingestion
- Drones submit flight plans via a streaming API (gRPC or WebSocket).
- Plans are validated and indexed asynchronously; the query endpoint reads from a read replica.
- Delta updates (e.g. rerouting) are processed as incremental conflict re-checks rather than full re-runs.

### 6.5 Algorithm Scalability
The O(M × N) quadratic minimisation per segment pair is inherently parallelisable. At scale, a query spawns a fan-out of parallel closest-approach computations across candidate segment pairs, collected by a reduce step that aggregates conflicts.

---

## 7. Limitations & Future Work

- **Wind model:** Constant speed assumed; real drones are affected by wind. A time-varying speed model would require numerical minimisation instead of the closed-form quadratic solution.
- **Uncertainty:** No probabilistic buffer around trajectories; a Gaussian uncertainty ellipsoid around each position could reduce false negatives caused by GPS drift.
- **Dynamic re-planning:** The system is strategic (pre-flight); a tactical layer responding to real-time deviations would complement it.
- **REST API:** No HTTP wrapper currently; wrapping `check_conflicts()` in a FastAPI endpoint would make this a deployable microservice.
- **Rerouting suggestions:** When a conflict is detected, automatically suggest a modified path that avoids it.
