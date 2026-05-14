# navsim-auto-research

Multi-agent auto-research pipeline targeting the [NAVSIM](https://github.com/autonomousvision/navsim) benchmark.

## Team
- **Pep** – Supervisor (planning, status, escalation)
- **Iniesta** – Research (SOTA scan, method selection)
- **Neymar** – Data (dataset acquisition, preprocessing)
- **Messi** – Train (training pipelines, hparam sweeps)
- **Suarez** – Eval (NAVSIM PDM scoring, submissions)
- **Xavi** – Ops (repo, CI, infra, wandb, leaderboard)

## Infra
- 8× RTX 5090 (32GB), shared host
- Data root: `/home/work/hanjianhua/data` (host-local, 2TB)
- W&B entity: `hanjianhua2012`, project: `navsim-auto-research`

## Status
Bootstrapping (Sprint 0). See `docs/STATUS.md`.

## Layout
```
configs/        # training & eval configs
scripts/        # one-shot bootstrap/download/launch scripts
docs/           # plans, status, decisions
results/        # leaderboard snapshots, eval logs
```
