# ARC-AGI-3

Workspace for the ARC Prize 2026 ARC-AGI-3 Kaggle competition.

This repository is currently in setup/documentation mode only. No solving
strategy, agent implementation, or experiment harness has been added yet.

Start here:

- [Competition guidelines](docs/competition_guidelines.md)
- [Research intake notes](knowledge/research_notes.md)
- Kaggle competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- ARC-AGI-3 docs: https://docs.arcprize.org/
- Official Kaggle starter: https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter

Local setup:

```bash
make setup
make verify-local
make notebook
```

`make submit` is intentionally separate. Before submitting, update
`notebooks/kernel-metadata.json` with the Kaggle notebook slug for this account.
