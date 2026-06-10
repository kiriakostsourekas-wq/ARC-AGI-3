# ARC-AGI-3 Research Intake Notes

Last updated: 2026-06-10.

This note summarizes the local papers in `knowledge/` plus the Medium post on
the ARC-AGI-3 Agent Preview winning solution. It is intended to guide future
engineering decisions, not to serve as a full literature review.

## Applicability Ranking

| Source | Applicability | How to Use It |
| --- | --- | --- |
| Dries Smit Medium post on StochasticGoose | High | Baseline strategy for ARC-AGI-3: simple RL, action-effect prediction, transition memory, hash de-duplication, level-local training. |
| GEAR: Genetic AutoResearch for Agentic Code Evolution | High for our research workflow | Use population/frontier search to manage agent variants and experiments; not necessarily part of the submitted runtime agent. |
| Deep Active Inference Agents | Medium-high | Borrow world-model + uncertainty + planning ideas; avoid full heavy active-inference implementation initially. |
| Human-inspired Episodic Memory for Infinite Context LLMs | Medium | Use event/episode segmentation and temporal-contiguous retrieval for trajectory memory; not direct KV-cache implementation unless we use an LLM agent. |
| Predictive Forgetting | Medium | Use as a principle for compressing/pruning experience buffers toward outcome-predictive state abstractions. |
| RLSR: Reinforcement Learning from Self Reward | Medium | Useful for self-improvement and generator-verifier thinking, but hosted/judge-LLM runtime is not allowed in Kaggle. |
| SEAL: Self-Adapting Language Models | Low-medium | Useful for offline adaptation/data generation ideas; runtime weight updates are risky under Kaggle time/packaging constraints. |
| Bottlenecked Transformers / Periodic KV Consolidation | Low initially | Mostly relevant only if we build a local LLM-based agent; conceptual support for memory consolidation. |

## Winning Preview Solution: StochasticGoose

Source: https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db

Key facts from the post:

- The ARC-AGI-3 preview winner used a practical RL-style approach rather than a
  direct LLM reasoning system.
- Directly putting many 64x64 frames into an LLM context was considered
  impractical because trajectories can be hundreds of steps, producing very
  large token histories.
- The core action-space problem was `ACTION6`, because it requires selecting
  coordinates in a 64x64 grid.
- The solution trained a CNN from frame observations to predict which simple
  actions were likely to change the game state.
- Coordinate actions used a spatially aware decoder rather than a flat 4096-way
  classifier.
- Transitions were stored permanently for off-policy training.
- Hash tables removed duplicate frame-action pairs.
- Experience buffers were reset between levels.
- The model was trained iteratively every few steps and reset between levels.
- The author notes frame segmentation as a promising improvement, reportedly
  used successfully by the second-place agent.
- Final preview score reported in the post was 12.58%, with the most levels and
  games completed among preview submissions.

Engineering relevance:

- This should be our first serious baseline once the starter project is in
  place.
- We should implement a state/action transition logger early, before clever
  policies.
- We should learn an action-effect predictor: "will this action change the
  frame/state?"
- For `ACTION6`, avoid naive uniform random clicking. Use spatial maps,
  connected components, object centers, frontier pixels, or a learned coordinate
  heatmap.
- Use frame hashing and transition de-duplication from the start.
- Keep level-local state and reset/adapt per level.
- Treat preview success as a baseline, not a complete solution. The full 2026
  competition may include games hardened against brute-force exploration.

## Paper Notes

### GEAR: Genetic AutoResearch for Agentic Code Evolution

Local file: `knowledge/GEAR_Autoresearch.pdf`

Core idea:

- Replace single-incumbent hill climbing with a bounded frontier of diverse
  elite research states.
- Expand the frontier through mutation and crossover.
- Track code changes, parentage, reflections, and performance statistics.
- Programmatic controllers outperformed prompt-only population management.

How it could help us:

- Highly useful for our experiment process.
- We should not keep only "the current best agent." We should preserve multiple
  branches: e.g. random/exploration-heavy, action-effect model, segmentation,
  simple model-based planner, local-LLM-assisted offline trainer.
- We can build a lightweight experiment registry with:
  - variant id,
  - parent ids,
  - mutation description,
  - games tested,
  - score/levels/actions,
  - failure notes,
  - artifacts/log paths.
- Later, we can recombine ideas: e.g. spatial click model from one variant plus
  memory compression from another.

Risk:

- The paper assumes an autonomous LLM research loop. We should apply the search
  discipline, not necessarily build a full autonomous coding agent.

### Deep Active Inference Agents

Local file: `knowledge/DeepInferenceAgents.pdf`

Core idea:

- Active inference combines perception and action via free-energy minimization.
- The paper uses learned latent dynamics, uncertainty, and Monte Carlo tree
  search to plan actions.
- It includes a habitual policy network to amortize planning in familiar states.

How it could help us:

- Useful as inspiration for a model-based ARC-AGI-3 agent.
- Build a learned or hand-engineered transition model for "what changes if I do
  action A?"
- Use uncertainty/surprise as intrinsic motivation for exploration.
- Use shallow search/planning over remembered transitions and simple predicted
  outcomes.
- Habitual policy idea maps well to ARC: once a level pattern is recognized,
  use cached action routines instead of planning from scratch.

Risk:

- The full deep active inference stack is too complex for a first implementation
  and may be too compute-heavy under Kaggle constraints.

### Human-inspired Episodic Memory for Infinite Context LLMs

Local file: `knowledge/EpisodicMemory.pdf`

Core idea:

- Segment long streams into event-like memory units using surprise and boundary
  refinement.
- Retrieve memory using both similarity and temporal contiguity.
- The method is designed for LLM long-context/KV memory, but the event-memory
  abstraction is broader.

How it could help us:

- Strong fit for trajectory memory in interactive games.
- Segment gameplay into events such as level start, state change, object moved,
  reward/progress change, repeated no-op region, failure, level completion.
- Retrieval should combine:
  - current-state similarity,
  - recent contiguous steps,
  - previously successful action fragments.
- Store event summaries rather than every raw frame forever.

Risk:

- Direct EM-LLM/KV-cache mechanics are not immediately useful unless the final
  agent includes a local LLM.

### Predictive Forgetting

Local file: `knowledge/Predictive_forgetting.pdf`

Core idea:

- Generalization improves when memory retains outcome-predictive information
  and discards incidental detail.
- Offline consolidation can iteratively compress high-fidelity experiences into
  more semantic/useful traces.

How it could help us:

- Provides a principled reason to compress the transition buffer.
- Keep raw transitions while a level is active, then consolidate into:
  - state hashes,
  - object/region features,
  - action effects,
  - successful sub-trajectories,
  - dead/no-op actions,
  - failure patterns.
- Use outcome-conditioned pruning: retain information predictive of level
  progress or state change; discard duplicate/noisy observations.

Risk:

- Mostly conceptual. We need simple engineering approximations, not
  information-theoretic machinery.

### RLSR: Reinforcement Learning from Self Reward

Local file: `knowledge/RLSR.pdf`

Core idea:

- LLMs can sometimes improve through self-judging, exploiting the gap between
  generating solutions and verifying them.
- The paper stresses reward-hacking fragility and the importance of judge
  quality/prompt design.

How it could help us:

- The generator-verifier gap is important for ARC-AGI-3: verifying action effect
  or progress is often easier than choosing the correct action.
- Prefer programmatic/local verifiers where possible:
  - did frame change?
  - did level/progress increase?
  - did available-action set change?
  - did state hash enter a new region?
  - did action sequence reach a previously promising state faster?
- Could use local LLM judges offline during development, but not hosted judges
  in Kaggle evaluation.

Risk:

- Hosted LLM judges are disallowed at runtime.
- Small/local self-judges may be brittle and reward-hackable.
- ARC gives sparse but real environment feedback, so self-reward should augment
  rather than replace environment signals.

### SEAL: Self-Adapting Language Models

Local file: `knowledge/SEAL.pdf`

Core idea:

- Models generate their own self-edits/training data and update directives.
- Reinforcement learning rewards self-edits that improve downstream performance.
- Includes experiments on knowledge incorporation and an ARC-AGI subset.

How it could help us:

- Useful for offline pretraining/adaptation workflows:
  - generate synthetic ARC-like tasks,
  - generate state/action transformation examples,
  - train small local models/adapters on transformed examples.
- The "self-edit" idea maps to generating better internal representations or
  curricula from raw trajectories.

Risk:

- Test-time gradient updates inside Kaggle may be too slow/fragile.
- Packaging trainable local models and adapters increases complexity.
- Treat as later-stage work after a strong non-LLM/RL baseline exists.

### Bottlenecked Transformers: Periodic KV Cache Consolidation

Local file: `knowledge/Periodic_KV_consolidation.pdf`

Core idea:

- Periodically rewrite/consolidate the transformer KV cache to improve
  reasoning/generalization.
- Uses memory consolidation/reconsolidation ideas and information bottleneck
  motivation.

How it could help us:

- If we use a local LLM, this supports periodically compressing/restructuring
  internal reasoning state rather than keeping raw frame history.
- Conceptually aligns with trajectory-event consolidation.

Risk:

- Direct implementation would require model internals/KV surgery and is not a
  first-pass competition tactic.

## Recommended First Engineering Direction

1. Start from the official starter kit and reproduce random/local play.
2. Build a transition logger and frame hashing/de-duplication.
3. Implement preview-winner-inspired baseline:
   - level-local experience buffer,
   - action-effect prediction,
   - spatial click candidate generation for `ACTION6`,
   - off-policy updates from stored transitions.
4. Add event segmentation:
   - no-op stretches,
   - surprise/frame-change boundaries,
   - object/region changes,
   - level progress/failure boundaries.
5. Add experiment registry inspired by GEAR:
   - keep multiple agent variants,
   - record parentage and mutations,
   - avoid losing partially useful ideas.
6. Only then evaluate heavier additions:
   - model-based planning,
   - intrinsic reward/curiosity,
   - local small model assistance,
   - offline synthetic task generation.

## Current Judgment

The Medium post should directly shape our first baseline. GEAR should shape our
research process. Episodic memory, predictive forgetting, and active inference
should shape our agent architecture incrementally. RLSR and SEAL are useful for
offline/self-improvement ideas but should not define the first Kaggle runtime.
Periodic KV consolidation is mostly future-facing unless we commit to a local
LLM architecture.

