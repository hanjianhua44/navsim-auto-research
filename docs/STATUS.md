# Project Status

Last updated: Sprint 0 kickoff.

## Sprint 0 – Bootstrap
| Owner | Task | State |
|---|---|---|
| Xavi | Init repo + skeleton + CI placeholder | doing |
| Neymar | Download NAVSIM **mini** to /home/work/hanjianhua/data | queued |
| Neymar | Probe sustained bandwidth, decide on full-set ETA | queued |
| Iniesta | Short-list 3–5 open-source autonomous-driving VLA candidates (high stars + recent) | queued |
| Messi | Stand by; read NAVSIM training entrypoints | queued |
| Suarez | Stand by; read NAVSIM PDM scoring code | queued |
| Pep | Run first stand-up once mini data lands | doing |

## Decisions log
- D0 (boss): Start with NAVSIM **mini** split; full set syncs in background.
- D0 (boss): Target = open-source AD VLA models with high stars + recent methods.
- D0 (Pep): Bots reply only when @-mentioned, so Pep drives the cadence.

## Sprint 0 — Decisions (Pep, after Iniesta's shortlist)
- **D1**: Baseline = **ReCogDrive-2B** (Plan A). DriveVLA-W0 推理作为可选对照（Plan B），先不强制。
- **D2**: 起步评测用 **NAVSIM v1 navtest mini**，pipeline 跑通 + PDMS 对齐 86/90 之后，再扩到 v2 `private_test_hard` leaderboard。
- **D3**: 第一周**只复现 Stage-2 IL**（目标 PDMS≈86），RL/GRPO 延到第二周。
- **D4**: sm_120 软件栈（torch 2.5+/CUDA 12.8 + flash-attn / xformers / vllm 兼容性）由 **Xavi** 先趟，是 S0 阻塞项。

## Sprint 1 — Reproduce ReCogDrive-2B IL (queued, 等 S0 出环境+数据)
| Owner | Task | State |
|---|---|---|
| Xavi | Trail 5090 sm_120 软件栈 (torch/flash-attn/xformers/vllm)，把可用版本钉进 env/setup_env.sh | doing |
| Neymar | 续跑 NAVSIM v1.1 mini-trainval + navtest mini，落到 /home/work/hanjianhua/data/navsim/ | doing |
| Suarez | 摸 PDMS / EPDMS 评测脚本，跑得通官方 ckpt 上的 mini eval | queued |
| Messi | 摸 ReCogDrive 三阶段训练 entrypoint，Stage-2 IL 配方 + LoRA 显存预估 | queued |
| Iniesta | 监督 baseline 适配过程；准备 Plan B（DriveVLA-W0 推理）作为对照 | standby |
