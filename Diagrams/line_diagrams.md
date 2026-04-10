# LINE_DIAGRAM 1 — Closest Approach Math (How the algorithm finds conflicts)
Shows: The quadratic minimization trick


PROBLEM: Two drones moving. Find their MINIMUM distance.

TIME ────────────────────────────────────────────────▶

t_overlap_start          t_overlap_end
     │                         │
     ▼                         ▼
     ├─────────────────────────┤
     │   shared time window    │


DRONE P position at time t:   P(t) = P_start + (t - t_start_P) × velocity_P
DRONE S position at time t:   S(t) = S_start + (t - t_start_S) × velocity_S

Relative displacement:
     D(t) = P(t) - S(t)
           = A + B × t          ← LINEAR in time!

Distance squared:
     dist²(t) = |D(t)|²
              = a×t² + b×t + c  ← QUADRATIC (parabola shape!)


    dist²
      │
      │      ╭──────╮
      │    ╭─╯      ╰─╮
      │  ╭─╯          ╰─╮
      │╭─╯              ╰─╮
      ├──────────────────────▶  time
      t_start    t_min    t_end
                   ↑
           MINIMUM DISTANCE
           (parabola bottom)
           t_min = -b / (2×a)


IF t_min is INSIDE the overlap window:
   → minimum distance = sqrt(a×t_min² + b×t_min + c)

IF t_min is OUTSIDE the window:
   → check the ENDPOINTS (t_start and t_end)
   → minimum is at whichever endpoint is closer


IF minimum distance < safety_buffer → CONFLICT!
IF minimum distance ≥ safety_buffer → SAFE!



# LINE_DIAGRAM 2 — How position_at_time() Works
Shows: Linear interpolation concept


MISSION: Drone flies from A(0,0) to B(100,0), speed=10 m/s
         Takes 10 seconds total.

Timeline:
t=0   t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9   t=10
 │     │     │     │     │     │     │     │     │     │     │
 ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
A──────○─────○─────○─────○─────○─────○─────○─────○─────○─────B
(0,0) (10,0)(20,0)(30,0)(40,0)(50,0)(60,0)(70,0)(80,0)(90,0)(100,0)


FORMULA:
  alpha = (t - t_start) / (t_end - t_start)
        = how far through the segment (0.0 = start, 1.0 = end)

  position = start + alpha × (end - start)


EXAMPLE: Where is drone at t=3?
  alpha = (3 - 0) / (10 - 0) = 0.3
  x = 0 + 0.3 × (100 - 0) = 30
  y = 0 + 0.3 × (0 - 0) = 0
  position = (30, 0, 0) ✓


MULTI-SEGMENT MISSION:
  (0,0) ──── (10,0) ──── (10,10) ──── (20,10)
    seg 0      │   seg 1    │    seg 2
   t=0→1       │   t=1→2   │   t=2→3

  At t=2.5 → we are in seg 2
  alpha = (2.5 - 2) / (3 - 2) = 0.5
  x = 10 + 0.5 × (20-10) = 15
  y = 10 + 0.5 × (10-10) = 10
  position = (15, 10, 0) ✓



# LINE_DIAGRAM 3 — Temporal Overlap Pre-filter
Shows: Why we skip segment pairs that don't overlap in time (performance trick)


CASE 1: NO OVERLAP — skip immediately, no math needed

  Seg P:  t=0 ──────── t=5
  Seg S:                     t=8 ──────── t=14
                        ↑
                     gap here
                  NO OVERLAP → SKIP


CASE 2: PARTIAL OVERLAP — do the math only for the shared window

  Seg P:  t=2 ──────────────── t=10
  Seg S:            t=6 ──────────────── t=14
                    ↑           ↑
               overlap_start  overlap_end
               (t=6)           (t=10)
                  DO MATH only between t=6 and t=10


CASE 3: ONE INSIDE THE OTHER — full inner segment is the overlap

  Seg P:  t=0 ────────────────────────── t=20
  Seg S:          t=5 ──────── t=10
                  ↑            ↑
             overlap_start   overlap_end
             DO MATH only between t=5 and t=10


WHY THIS MATTERS:
  Without this filter: check ALL segment pairs (slow)
  With this filter:    skip pairs flying at different times (fast)
  Result: much faster for large missions with many waypoints



# LINE_DIAGRAM 4 — Safety Buffer Concept
Shows: What "safety buffer" means and when a CONFLICT is triggered


TOP VIEW (looking down from above):

  Drone A ●                              ● Drone B
          ←──────────── 25m ─────────────→
          [buffer = 8m]  →  25m > 8m  →  SAFE ✓


  Drone A ●          ● Drone B
          ←── 5m ────→
          [buffer = 8m]  →  5m < 8m  →  CONFLICT ✗


  Drone A ●    ● Drone B
          ← 0m →
          [buffer = 8m]  →  0m < 8m  →  CONFLICT ✗  (collision!)


SAFETY BUFFER = minimum allowed gap between any two drones at any time
  - Default in this system: 8 metres (can be changed)
  - Checked continuously along entire flight path, not just at waypoints
  - Even a single moment of violation = CONFLICT reported



# LINE_DIAGRAM 5 — Feasibility Check
Shows: Whether the drone can complete its mission within the allowed time window


CASE 1: FEASIBLE ✓

  Mission needs:  ├──────────── 10s ────────────┤
  Time window:    ├────────────────── 15s ───────────────┤
                                      ↑
                           fits inside → FEASIBLE ✓


CASE 2: INFEASIBLE ✗

  Mission needs:  ├──────────────────── 14s ────────────────────┤
  Time window:    ├────────── 10s ──────────┤
                                            ↑
                                  overflows → INFEASIBLE ✗
                                  mission cannot be completed in time


HOW IT IS CALCULATED:
  mission_duration = total distance / speed
                   = sum of all segment lengths / drone speed

  Example:
    Waypoints: (0,0) → (50,0) → (50,100)
    Segment 1 length = 50m
    Segment 2 length = 100m
    Total distance   = 150m
    Speed            = 10 m/s
    Duration         = 150 / 10 = 15 seconds

    If mission_end = 20s → 15 < 20 → FEASIBLE ✓
    If mission_end = 12s → 15 > 12 → INFEASIBLE ✗
