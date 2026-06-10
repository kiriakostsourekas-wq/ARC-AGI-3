# ARC Prize 2026 ARC-AGI-3 Guidelines

Last checked: 2026-06-10.

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

## Paper-Derived Benchmark Constraints

The ARC-AGI-3 paper frames the benchmark as a test of agentic intelligence,
not static puzzle solving. Important design constraints from the paper:

- Agents receive no explicit objective or natural-language instructions. They
  must infer mechanics and win conditions through interaction.
- Each environment is turn-based. At each turn, the agent receives a frame or
  frame sequence and must choose one action; the environment does not change
  asynchronously between actions.
- Observations are 64x64 grids where each cell is one of 16 colors. Frame
  sequences may represent non-interactive transition animations.
- The action space for each environment is a game-specific subset of five key
  actions, undo, and coordinate selection on the 64x64 grid.
- The benchmark is designed around Core Knowledge priors only: objectness,
  geometry/topology, basic physics, and agentness. Environments avoid language,
  numbers, letters, recognizable real-world icons, and cultural conventions.
- Public environments are demonstration interfaces, not a comprehensive
  training distribution. The paper reports 25 public demonstration
  environments, 55 semi-private environments, and 55 fully private environments.
- The private set is intentionally harder and out-of-distribution relative to
  the public set, with broader mechanics and deeper compositional reasoning.
- First-run efficiency matters. The paper's human baseline is the second-best
  first-run human playthrough, so exploration cost is part of the score.
- Context and history management are central challenges because naive storage
  of raw 64x64 frame histories grows quickly.
- The paper describes StochasticGoose as a CNN/RL action-change predictor that
  encoded 64x64 frames with a four-layer convolutional network, scored 12.58%
  in the preview competition, and completed 18 levels.

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

## Model And Dependency Constraints

There is no official whitelist of allowed model names. The practical rule is
that the submitted agent must run fully inside the offline Kaggle notebook.

Allowed for submitted runs:

- A non-ML or classical search/programmatic agent.
- Any local/open-weight model that is already available to the notebook at
  evaluation time.
- Model weights attached as Kaggle input data/model artifacts, if their license
  allows our intended use and prize-publication obligations.
- Models trained by us before submission, provided the resulting weights and
  inference code are packaged for offline execution.
- Local test-time compute inside the Kaggle runtime budget.

Not allowed for submitted runs:

- Runtime API calls to hosted models such as GPT, Claude, Gemini, hosted
  Hugging Face endpoints, or other external services.
- Downloading model weights, Python packages, data, prompts, or tools from the
  internet during the Kaggle evaluation run.
- Any architecture that depends on network access, a private server, a remote
  database, or a cloud inference endpoint.

Engineering implications:

- Treat the Kaggle submission as an offline appliance: all code, dependencies,
  weights, config, and prompt assets must be present before the rerun starts.
- Do not design the final agent around online LLM calls. We can use external
  tools while researching locally, but the submitted runtime path must not call
  them.
- Keep a separate path for offline inference and make it easy to smoke-test
  with networking disabled.
- Track model licenses before adding weights. Prize eligibility requires
  reproducible open-source submissions and public sharing of authored methods.
- Watch size and runtime: current metadata reports a `20480` MB submission size
  limit and `540` minute CPU/GPU runtime limits.
- Prefer lightweight models or deterministic/search-heavy methods until there
  is evidence that a larger local model improves score enough to justify the
  packaging and runtime cost.

## Scoring

ARC-AGI-3 uses Relative Human Action Efficiency (RHAE).

The score rewards:

- Completion: how many levels the agent completes in each game.
- Efficiency: how many environment-changing actions it takes compared with a
  human baseline.

Internal computation does not count as an action. Only discrete interactions
that affect the environment count.

The paper emphasizes that this penalizes brute force: a system that blindly
tries many actions is scored worse than a system that forms a model of the
environment and plans efficient actions. The official human baseline is based
on first-run human attempts, so agents are rewarded for efficient adaptation on
first contact, not just eventual completion.

Per completed level:

```text
level_score = (human_baseline_actions / ai_actions) ^ 2
```

Additional scoring details:

- Level score is capped at `1.15`.
- Per-game score is a weighted average of level scores.
- Later levels have higher weight because level number is used as the weight.
- Later levels also matter conceptually because the paper designs environments
  so later levels require accumulating and integrating mechanics learned in
  earlier levels.
- Total score is the average of all game scores.
- Completing only early levels caps the maximum possible game score even if
  those levels are solved efficiently.

## Action Interface

All games expose a standardized action interface. Per the paper, each game
uses a subset of these controls so the difficulty is in reasoning about the
environment, not learning a complex control scheme:

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
- ARC-AGI-3 paper v1:
  https://arxiv.org/pdf/2603.24621v1
- ARC-AGI-3 paper landing page:
  https://arxiv.org/abs/2603.24621
- Kaggle StochasticGoose sample:
  https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose
- StochasticGoose source reference:
  https://github.com/DriesSmit/ARC3-solution
