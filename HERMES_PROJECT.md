# FoveaEdge
## OpenVINO Event-Driven Foveated Edge Vision Engine

> Research project: dynamically allocate high-resolution inference only where visual information requires it, while maintaining predictable edge-device performance.

---

# 1. MISSION

Build FoveaEdge as a serious OpenVINO research/engineering project.

The objective is NOT:

- another YOLO demo
- another object-detection application
- another OpenVINO wrapper
- a generic video analytics dashboard

The objective is to investigate and implement a new execution strategy for high-resolution edge vision:

    LOW-COST PERIPHERAL ANALYSIS
              ↓
        EVENT DETECTION
              ↓
       ROI GENERATION
              ↓
       ROI PRIORITIZATION
              ↓
      HARDWARE-AWARE SCHEDULING
              ↓
     ASYNCHRONOUS OpenVINO
              ↓
       HIGH-RES INFERENCE
              ↓
       RESULT FUSION
              ↓
        FULL-FRAME OUTPUT

The central research question is:

> Can spatially selective, event-driven inference reduce computational cost for high-resolution edge vision while preserving small-object detection performance?

---

# 2. IMPORTANT RESEARCH POSITION

Do NOT claim:

> "OpenVINO has never done this."

Do NOT claim:

> "Nobody has ever invented foveated inference."

Do NOT claim:

> "This is the world's first system."

Those claims must only be made after extensive investigation.

Instead, establish exactly what exists.

The project's contribution should be based on measurable engineering differences.

The investigation should specifically compare:

1. Existing OpenVINO samples
2. Existing OpenVINO notebooks
3. Existing OpenVINO scheduling APIs
4. Existing dynamic-shape functionality
5. Existing asynchronous inference functionality
6. Existing edge-video pipelines
7. Existing foveated vision research
8. Existing ROI inference systems
9. Existing dynamic-resolution systems
10. Existing adaptive inference systems

The final README must clearly distinguish:

EXISTING TECHNOLOGY

from

OUR IMPLEMENTATION

from

OUR EXPERIMENTAL CONTRIBUTION.

---

# 3. CORE IDEA

Traditional high-resolution edge vision usually follows:

4K FRAME
   ↓
RESIZE
   ↓
640×640
   ↓
DEEP LEARNING
   ↓
OBJECTS

The problem is that resizing the entire image destroys spatial information.

FoveaEdge investigates:

4K FRAME
   │
   ├──► LOW-COST PERIPHERAL ANALYSIS
   │
   │        detects areas of interest
   │
   ▼
EVENT MAP
   │
   ▼
ROI GENERATOR
   │
   ▼
ROI PRIORITIZATION
   │
   ▼
HIGH-RES CROPS
   │
   ▼
OpenVINO INFERENCE
   │
   ▼
COORDINATE FUSION
   │
   ▼
4K RESULT

Most of the frame should not require expensive neural inference.

The expensive computation should be concentrated where it provides the most information.

---

# 4. THE ACTUAL INVENTION TO INVESTIGATE

The main research contribution is NOT simply:

"crop the moving areas."

That is too weak.

The important component is the combination of:

## Event-driven spatial allocation

+

## ROI quality/importance scoring

+

## Hardware-aware scheduling

+

## asynchronous inference

+

## computational budget control

+

## temporal consistency

The system should continuously decide:

> Which regions deserve expensive inference right now?

This decision should depend on:

- visual activity
- object likelihood
- ROI size
- ROI overlap
- historical importance
- temporal persistence
- inference queue pressure
- available compute
- latency budget
- maximum ROI budget

---

# 5. FOVEA SCHEDULER

Create a custom scheduler.

Each candidate ROI receives an importance score.

Example conceptual score:

    Score =
        α * motion
      + β * persistence
      + γ * object_likelihood
      + δ * novelty
      - ε * area
      - ζ * overlap
      - η * compute_cost

The exact formulation must be experimentally developed rather than blindly copied.

The scheduler should answer:

    Which ROI should be processed?

    At what resolution?

    When should it be processed?

    How many ROIs can be processed?

    Should several ROIs be merged?

    Should an ROI be skipped?

    Should the system fall back to global inference?

---

# 6. COMPUTATIONAL BUDGET

FoveaEdge should operate under a configurable compute budget.

Example:

    MAX_ROIS = 4

    TARGET_LATENCY = 33 ms

    TARGET_FPS = 30

If too many candidate regions appear:

    candidate ROIs
          ↓
    score regions
          ↓
    rank regions
          ↓
    select highest-value regions
          ↓
    merge compatible regions
          ↓
    discard low-value regions

The system must never blindly process every ROI.

---

# 7. TEMPORAL MEMORY

A major development direction is temporal consistency.

Instead of treating every frame independently:

Frame N:
    ROI A detected

Frame N+1:
    ROI A predicted

Frame N+2:
    ROI A tracked

Frame N+3:
    ROI A updated

This can reduce repeated expensive peripheral analysis and unnecessary inference.

The scheduler should eventually maintain state for each ROI:

- position
- velocity
- confidence
- age
- last inference timestamp
- priority
- predicted position

---

# 8. OPENVINO ROLE

OpenVINO is central to the project.

Use official OpenVINO capabilities wherever appropriate.

Investigate and use:

- OpenVINO Runtime
- model conversion
- dynamic shapes
- asynchronous inference
- AsyncInferQueue
- device selection
- CPU execution
- GPU execution where available
- NPU execution where available
- AUTO/MULTI where appropriate
- performance hints
- model caching
- NNCF quantization
- INT8 inference
- profiling tools

Do not invent APIs.

Use official APIs and document why they are used.

---

# 9. HARDWARE

Primary target:

Intel hardware.

Test hardware available to the developer.

Potential targets include:

- Intel CPU
- Intel integrated GPU
- Intel Arc GPU
- Intel Core Ultra NPU

Do not assume a device exists.

Detect it.

Record the actual devices reported by OpenVINO.

Example:

    CPU
    GPU
    NPU

If GPU/NPU hardware is unavailable, CPU experiments must still work.

---

# 10. BASELINES

FoveaEdge must NOT be evaluated in isolation.

Create at least three baselines.

## Baseline A — Global Low Resolution

4K
 ↓
640×640
 ↓
model
 ↓
result

Measure:

- FPS
- latency
- memory
- accuracy
- small-object recall

---

## Baseline B — Full Resolution

4K
 ↓
model
 ↓
result

Measure:

- FPS
- latency
- memory
- accuracy
- compute cost

This establishes the expensive high-quality baseline.

---

## Baseline C — FoveaEdge

4K
 ↓
peripheral analysis
 ↓
selected ROIs
 ↓
high-resolution inference
 ↓
fusion

Measure exactly the same metrics.

---

# 11. SUCCESS CRITERIA

Never fabricate targets as results.

Targets are hypotheses.

Investigate whether FoveaEdge can achieve:

- lower latency
- higher throughput
- lower compute utilization
- lower energy consumption
- comparable small-object accuracy

A result that fails the target is still valuable if it explains why.

---

# 12. BENCHMARK METRICS

Record:

## Performance

- FPS
- frame latency
- inference latency
- preprocessing latency
- ROI generation latency
- scheduling latency
- queue waiting time

## Accuracy

- mAP
- small-object recall
- precision
- missed detections

## Efficiency

- CPU utilization
- GPU utilization
- NPU utilization
- memory
- power where measurable

## Scheduler

- average ROI count
- selected ROI count
- rejected ROI count
- merged ROI count
- average ROI area
- inference frequency

---

# 13. MOST IMPORTANT EXPERIMENT

The central experiment should answer:

> How much computation can be removed before small-object detection quality becomes unacceptable?

Create a compute/accuracy curve.

For example:

    1 ROI
    2 ROIs
    3 ROIs
    4 ROIs
    6 ROIs
    8 ROIs

Compare:

    computational cost
           vs
    detection quality

This experiment is more important than simply claiming:

"FoveaEdge is faster."

---

# 14. DEMO

Create a professional demonstration.

Display:

LEFT:

Original high-resolution frame.

CENTER:

Detected active/foveal regions.

RIGHT:

High-resolution ROI inference.

OVERLAY:

FPS
Latency
ROI count
Queue pressure
CPU
GPU
NPU
Compute budget

Also provide a button to switch between:

    GLOBAL LOW-RES
    FULL-RES
    FOVEAEDGE

The differences should be visually obvious.

---

# 15. PROJECT STRUCTURE

Use:

FoveaEdge/

├── src/
│   ├── capture/
│   ├── peripheral/
│   ├── roi/
│   ├── scheduler/
│   ├── inference/
│   ├── fusion/
│   ├── tracking/
│   ├── telemetry/
│   └── ui/
│
├── models/
│
├── benchmark/
│
├── tests/
│
├── experiments/
│
├── docs/
│
├── assets/
│
├── scripts/
│
├── README.md
├── HERMES_PROJECT.md
├── pyproject.toml
└── LICENSE

---

# 16. DEVELOPMENT PHILOSOPHY

Do not build everything at once.

Every component must first have a measurable experiment.

Example:

Before building scheduler:

    implement ROI candidates
    generate test data
    measure candidate quality

Then:

    implement scheduler
    benchmark scheduler

Then:

    integrate OpenVINO

Then:

    benchmark complete pipeline

---

# 17. 20-DAY PLAN

## DAY 1

Repository + environment.

Verify:

- Python
- OpenVINO
- OpenCV
- Git
- available OpenVINO devices

Create reproducible environment.

Commit:

    day-01: initialize research environment

---

## DAY 2

OpenVINO model loading.

Create minimal inference pipeline.

Verify real inference.

Commit:

    day-02: establish OpenVINO inference baseline

---

## DAY 3

High-resolution video ingestion.

Build measurement framework.

Commit:

    day-03: add high-resolution video benchmark harness

---

## DAY 4

Global low-resolution baseline.

Measure actual performance.

Commit:

    day-04: establish low-resolution baseline

---

## DAY 5

Full-resolution baseline.

Measure actual performance.

Commit:

    day-05: establish full-resolution baseline

---

## DAY 6

Peripheral event detection.

Implement:

- frame difference
- thresholding
- morphology
- contour extraction

Commit:

    day-06: implement peripheral event detection

---

## DAY 7

ROI generation.

Implement candidate generation and filtering.

Commit:

    day-07: implement event-driven ROI generation

---

## DAY 8

ROI merging/clustering.

Develop and benchmark merging strategy.

Commit:

    day-08: add adaptive ROI clustering

---

## DAY 9

ROI importance scoring.

Implement first scheduler.

Commit:

    day-09: implement ROI priority scheduler

---

## DAY 10

Foveal inference.

Connect selected ROIs to OpenVINO.

Commit:

    day-10: integrate selective ROI inference

---

## DAY 11

Coordinate projection.

Map ROI detections back to full-frame coordinates.

Commit:

    day-11: implement coordinate fusion

---

## DAY 12

Async inference.

Integrate AsyncInferQueue.

Commit:

    day-12: add asynchronous inference pipeline

---

## DAY 13

Compute budget controller.

Introduce latency/ROI limits.

Commit:

    day-13: implement compute budget controller

---

## DAY 14

Temporal consistency.

Add ROI history/tracking.

Commit:

    day-14: add temporal ROI persistence

---

## DAY 15

Accuracy benchmark.

Compare all approaches.

Commit:

    day-15: add accuracy evaluation framework

---

## DAY 16

Hardware experiments.

Test available:

- CPU
- GPU
- NPU

Commit:

    day-16: benchmark Intel execution devices

---

## DAY 17

Performance optimization.

Profile bottlenecks.

Optimize only measured bottlenecks.

Commit:

    day-17: optimize measured pipeline bottlenecks

---

## DAY 18

Professional visualization.

Build telemetry dashboard.

Commit:

    day-18: add real-time performance visualization

---

## DAY 19

Research-quality documentation.

Add:

- architecture diagram
- methodology
- benchmarks
- limitations
- related work
- reproducibility instructions

Commit:

    day-19: document research methodology and results

---

## DAY 20

Final experiment.

Run complete benchmark.

Generate:

- tables
- graphs
- demo video
- final report

Commit:

    day-20: publish final FoveaEdge evaluation

Create GitHub release.

---

# 18. GITHUB DISCIPLINE

Every day must contain REAL work.

Do not create fake commits simply to generate contribution-graph activity.

Before every push:

1. git status
2. inspect diff
3. run tests
4. verify no secrets
5. verify documentation
6. commit
7. push

Never commit:

- API keys
- passwords
- tokens
- `.env`
- private datasets
- credentials

Use `.gitignore`.

---

# 19. DAILY REPORT

Create:

docs/progress/day-XX.md

Each report must contain:

# Day XX

## Objective

## Work Completed

## Files Changed

## Experiments

## Actual Results

## Problems

## Decisions

## Next Step

## Commit

The report must contain REAL information only.

---

# 20. AUTONOMOUS HERMES MODE

Hermes is authorized to work autonomously INSIDE THIS REPOSITORY.

Hermes may:

- create files
- modify project files
- run experiments
- run tests
- install project-local dependencies
- update documentation
- commit completed milestones
- push completed milestones

Hermes must NOT:

- fabricate results
- fabricate novelty
- expose credentials
- delete Git history
- force-push
- modify unrelated repositories
- make destructive system changes
- push broken work solely for GitHub activity
- claim an OpenVINO feature is new without investigation

If a critical decision requires unavailable hardware, missing credentials, destructive system changes, or external authorization, stop and document the blocker.

---

# 21. DAILY START PROCEDURE

When Hermes starts inside this repository:

1. Read HERMES_PROJECT.md.
2. Inspect git status.
3. Inspect current milestone.
4. Read the latest progress report.
5. Inspect existing implementation.
6. Determine the current day.
7. Work ONLY on the current milestone.
8. Run tests.
9. Record actual results.
10. Commit.
11. Push.
12. Write the daily report.

Never repeat completed work unnecessarily.

Never skip ahead without completing the current milestone.

---

# 22. RESEARCH INTEGRITY

The project's reputation depends on technical honesty.

If an experiment shows:

    FoveaEdge is slower

record it.

If:

    dynamic shapes are slower

record it.

If:

    ROI generation becomes the bottleneck

record it.

If:

    GPU/NPU execution does not help

record it.

A negative result is scientifically useful.

The goal is not to manufacture a "breakthrough."

The goal is to discover whether the architecture actually produces a measurable advantage.

---

# 23. FINAL OUTCOME

At Day 20 the project should ideally provide:

1. Working OpenVINO pipeline
2. Reproducible benchmark
3. Baseline comparison
4. FoveaEdge scheduler
5. Async inference
6. Hardware measurements
7. Accuracy evaluation
8. Performance graphs
9. Professional demo
10. Engineering documentation
11. Research report
12. Clean GitHub history

The final project should be strong enough to demonstrate:

- OpenVINO expertise
- computer vision knowledge
- systems engineering
- asynchronous programming
- hardware-aware optimization
- benchmarking
- research methodology
- Git/GitHub engineering discipline

---

# FINAL PRINCIPLE

Do not optimize for the appearance of innovation.

Optimize for:

    REAL PROBLEM
        +
    REAL ENGINEERING
        +
    REAL EXPERIMENTS
        +
    REAL MEASUREMENTS
        +
    REAL OPENVINO USAGE
        +
    REPRODUCIBLE RESULTS

That is what should make FoveaEdge recognizable.