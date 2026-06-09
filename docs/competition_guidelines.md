# ARC Prize 2026 ARC-AGI-3 Guidelines

Last checked: 2026-06-09.

This file captures the competition facts we have verified so far. It is not a
solver design document.

## Core Competition

- Track: ARC Prize 2026 - ARC-AGI-3.
- Goal: build an AI agent that can interact with novel, instruction-free
  environments and solve them efficiently.
- Benchmark type: interactive reasoning, not the older static ARC grid format.
- Required capabilities: exploration, modeling, goal-setting, planning, and
  execution.
- Prize pool for ARC-AGI-3: USD 850,000 total.
- Prize eligibility requires open-sourcing the solution code and methods.

## Key Dates

- Competition start: 2026-03-25.
- ARC-AGI-3 milestone 1: 2026-06-30.
- ARC-AGI-3 milestone 2: 2026-09-30.
- Team merger deadline in Kaggle metadata: 2026-10-26 23:59 UTC.
- New entrant deadline in Kaggle metadata: 2026-10-26 11:59 UTC.
- Final submission deadline: 2026-11-02 23:59 UTC.
- Results announced: 2026-12-04.

## Current Kaggle Metadata

Checked through Kaggle's competition metadata endpoint for
`arc-prize-2026-arc-agi-3`.

- Competition id: `133468`.
- Required submission filename: `submission.parquet`.
- Required row id column name: `row_id`.
- Submission type: kernel-only submissions (`onlyAllowKernelSubmissions: true`).
- Daily official submission limit: `maxDailySubmissions: 1`.
- Scored/final submissions reported by metadata: `numScoredSubmissions: 2`.
- Public leaderboard percentage: `50`.
- Max team size: `8`.
- Requires identity verification: `true`.
- CPU runtime limit: `540` minutes.
- GPU runtime limit: `540` minutes.
- Submission size limit: `20480` MB.
- Evaluation metric name: `ARC-AGI-3 Metric`.

Note: the daily submission field is a maximum daily submission limit. The
current numeric value is `1`.

## Submission Workflow

The official ARC-AGI-3 Kaggle starter is designed as a local development loop:

1. Accept the Kaggle competition rules.
2. Use Python 3.12.
3. Store the Kaggle API token in a project-local `.kaggle/access_token` file.
4. Run `make setup`.
5. Edit `agent/my_agent.py`.
6. Run local checks with `make play-local` or `make verify-local`.
7. Build and push the Kaggle notebook with `make submit`.
8. Run `make status`.
9. Once the Kaggle notebook run is complete, open the notebook on Kaggle and
   submit the generated `submission.parquet` output file to the competition.

The starter documents `agent/my_agent.py` as the normal edit surface. It defines
a `MyAgent` class with:

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        ...

    def choose_action(self, frames, latest_frame) -> GameAction:
        ...
```

The starter handles notebook packaging, Kaggle plumbing, and generation of the
`submission.parquet` artifact.

## Competition Mode Constraints

Kaggle forces ARC-AGI-3 into competition mode. In this mode:

- Environments are interacted with through the API.
- Scoring is against all available environments, including environments the
  agent does not choose to interact with.
- Only level resets are allowed; game resets become level resets.
- The agent can call `make` only once per environment.
- Only one scorecard can be opened.
- In-flight scorecard scoring is unavailable; `get_scorecard` does not work
  during the run.

## Evaluation Environment

- No internet access is available during Kaggle evaluation.
- API-based systems such as hosted LLM calls are therefore not usable in the
  evaluation run.
- Accelerated Kaggle sessions also have internet disabled in the starter
  workflow.
- The official starter defaults to a T4 GPU notebook, but it can be changed to
  `cpu`, `t4`, `p100`, or `rtx6000` in `scripts/build_notebook.py`.

## Scoring

ARC-AGI-3 uses Relative Human Action Efficiency (RHAE).

The score rewards:

- Completion: how many levels the agent completes in each game.
- Efficiency: how many environment-changing actions it takes compared with a
  human baseline.

Internal computation does not count as an action. Only discrete interactions
that affect the environment count.

Per completed level:

```text
level_score = (human_baseline_actions / ai_actions) ^ 2
```

Additional scoring details:

- Level score is capped at `1.15`.
- Per-game score is a weighted average of level scores.
- Later levels have higher weight because level number is used as the weight.
- Total score is the average of all game scores.
- Completing only early levels caps the maximum possible game score even if
  those levels are solved efficiently.

## Action Interface

All games expose a standardized action interface:

- `RESET`: initialize or restart the game/level state.
- `ACTION1`: simple action, semantically mapped to up.
- `ACTION2`: simple action, semantically mapped to down.
- `ACTION3`: simple action, semantically mapped to left.
- `ACTION4`: simple action, semantically mapped to right.
- `ACTION5`: simple action such as interact/select/rotate/execute, depending
  on the game.
- `ACTION6`: complex action requiring `x,y` coordinates in the `0-63` range.
- `ACTION7`: simple undo action when supported.

Each game exposes the currently available actions in frame metadata. If
`ACTION6` is available, the metadata only reports availability, not valid active
coordinates.

When a game reaches game-over state, only `RESET` is valid. Sending another
action in game-over state can produce a `400 Bad Request`.

## Open Source And Licensing

ARC Prize states that prize eligibility requires reproducible, open-source
submissions. Authored code and methods must be made open source under a
permissive public-domain-style license such as `CC0` or `MIT-0`; third-party
code must at least allow public sharing.

Current repo note: the GitHub repository already contains an MIT `LICENSE`.
Before making prize-eligible submissions, confirm whether MIT is acceptable for
this track or switch/add the exact license ARC requests for authored code.

## Sources

- Kaggle competition:
  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Kaggle metadata endpoint:
  https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=arc-prize-2026-arc-agi-3
- ARC Prize 2026 ARC-AGI-3 page:
  https://arcprize.org/competitions/2026/arc-agi-3
- ARC Prize 2026 overview and rules:
  https://arcprize.org/competitions/2026
- ARC-AGI-3 Kaggle starter docs:
  https://docs.arcprize.org/arc-prize-2026
- ARC-AGI-3 scoring methodology:
  https://docs.arcprize.org/methodology
- ARC-AGI-3 actions:
  https://docs.arcprize.org/actions
- ARC-AGI-3 competition mode:
  https://docs.arcprize.org/toolkit/competition_mode

