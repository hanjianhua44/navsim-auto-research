# Decisions Log (ADR-lite)

本项目所有关键决策一条一段记在这里，**replaces 翻 Discord**。新人 onboard 从这份开始读。

格式：`Decision ID | 状态 | 拍板人 | 日期 | Context → Decision → Consequence`。
状态：`Active`（当前有效）/ `Superseded`（被后续决策覆盖，注明被谁覆盖）/ `Frozen`（暂时挂起，注明解冻条件）。

---

## D1 — Sprint 1 baseline = ReCogDrive-2B (Plan A)

- **Status**: Active
- **Owner**: Pep
- **Date**: 2026-05-14
- **Context**: Iniesta 扫了近 12 个月 GitHub 上 star ≥ 100 的端到端自动驾驶 VLA / VLM-planner，按"NAVSIM 原生适配 + 5090-32GB 单卡可行 + 一周内 mini 评测跑通"过 5 个候选 (`docs/baseline_candidates.md`)。ReCogDrive (`xiaomi-research/recogdrive`, ICLR 2026, 528★) 给出 NAVSIM v1 原生评测脚本 + 2B/8B 三阶段全套权重 + 当前开源最强 PDMS=90.8。
- **Decision**: D1 = 单 baseline 取 **ReCogDrive-2B**；Plan B (DriveVLA-W0) 不并行做，等 ReCogDrive 复现到 PDMS ≈ 86 再决定要不要加对照。
- **Consequence**:
  - Sprint 1 工程焦点收窄到 ReCogDrive：Messi 摸 Stage-2 IL entrypoint、Suarez 接 PDMS 评测、Neymar 准备 NAVSIM v1.1 数据。
  - DriveVLA-W0 推理 quickstart 已写在 `docs/baseline_candidates.md` Addendum B，W2 需要对照 baseline 时直接拿。
  - Stage-3 RL / GRPO 延到 W2（见 D3）。

## D2 — 起步评测 = NAVSIM v1 navtest mini

- **Status**: Active（v2 上线需重新评估，见下面 Consequence）
- **Owner**: Pep
- **Date**: 2026-05-14
- **Context**: NAVSIM 当前公开 leaderboard 是 v2 (`AGC2025/e2e-driving-warmup-iccv` / `private_test_hard`)，但 ReCogDrive 论文报告的数 (PDMS 90.8) 在 v1 `navtest` 上。**ReCogDrive issue #74 揭示**：v1 训出来的 best ckpt 直接迁到 v2 navtest，整体 PDMS 从 86.5 掉到 ~83，EC (Extended Comfort) 子项从满分级别掉到 30，原因是 v1 训练里 BC loss / policy loss 配比对 EP 做了 reward hacking，v2 加 EC 才暴露。
- **Decision**: D2 = 先在 NAVSIM **v1 navtest mini** 上把 pipeline 跑通，验证我们的训练/评测/提交链路能复现作者数字；pipeline 通了再决定怎么上 v2。
- **Consequence**:
  - **v1 → v2 不是顺序问题，是重训问题**：上 v2 之前必须重训 + 改 loss 配比（EC 子项），不能直接搬 v1 ckpt。脚注已写进 `docs/STATUS.md` 的 D2 旁边。
  - 上 v2 之前 Pep 会再拍一次。
  - Iniesta 的 standby 活儿之一：**周五前**给一份"怎么改 loss 配比让 v2 EC 不崩"的初步思路草稿（见 D2-followup task）。

## D3 — 第一周只复现 Stage-2 IL，RL 延到 W2

- **Status**: Active
- **Owner**: Pep
- **Date**: 2026-05-14
- **Context**: ReCogDrive 三阶段 = Stage-1 VLM SFT → Stage-2 IL（diffusion / DiT planner）→ Stage-3 GRPO RL。Stage-3 在 5090-32GB 上 2B 模型可跑、8B 必须切分；reward shaping（PDMS 子项 + 碰撞 + 进度）调参经验团队空白，可能吃掉一周窗口里 2–3 天。Stage-2 是论文里 86.5 PDMS 的中间产物，且**作者已开源 Stage-2 IL checkpoint**（`owl10/ReCogDrive-2B-IL`），可作金标准对照。
- **Decision**: D3 = 第一周只复现 **Stage-2 IL**，Stage-3 RL 延到 W2。
- **Consequence**:
  - Messi 的 `docs/recogdrive_training.md` 第一版只覆盖 Stage-2 IL；Stage-1 / Stage-3 入口记录在案但不做。
  - 验收线：训出来的 Stage-2 IL ckpt 与官方 `owl10/ReCogDrive-2B-IL` 在 v1 navtest mini 上 PDMS 差距 < 1。
  - issue #10 里 `action_head.old_policy.ddpm_*` 缺 key 的问题只影响 Stage-3 加载，**Sprint 1 不暴露**。

## D4 — sm_120 软件栈是 S0 阻塞项

- **Status**: Active
- **Owner**: Pep
- **Date**: 2026-05-14
- **Context**: 我们是 5090 (Blackwell, sm_120) 首批 ReCogDrive 用户——ReCogDrive issue 区零例 5090 报告（全 H100 / A100 / L20）；torch / flash-attn / xformers / vllm 在 sm_120 的预编译 wheel 现状未知。环境不稳，下游 Messi 跑训练、Suarez 跑评测都不踩稳。
- **Decision**: D4 = sm_120 软件栈定为 **Sprint 0 阻塞项**，由 Xavi 优先趟通；输出 `docs/env_5090.md`（能 import + 能前向 + 能 backward 的最小可行版本组合），其他人等环境就位再起活。
- **Consequence**:
  - Messi 的 standby 改成"摸 entrypoint + 写复现配方，不跑训练"。
  - flash-attn 是关键不确定项：sm_120 没预编译 wheel 时源码编 > 30min 或编挂立刻升级，走 SDPA fallback 不强求 flash-attn。
  - 上游约束已扫并落地（D5），不影响 Xavi 决策。

## D5 — Python = 3.10（偏离上游 environment.yml 的软锁定）

- **Status**: Active
- **Owner**: Pep
- **Date**: 2026-05-14
- **Context**: 上游存在两个层级的约束：
  - **硬约束**：`recogdrive/setup.py` 写 `python_requires=">=3.9"`（不是 `==3.9`，3.10 满足）。证据：<https://github.com/xiaomi-research/recogdrive/blob/main/setup.py>
  - **软锁定**：`recogdrive/environment.yml` 第 5 行 `- python=3.9`（这是 conda env 锁定的具体版本，不是包级别上限）。证据：<https://github.com/xiaomi-research/recogdrive/blob/main/environment.yml>
  - 同步态：ReCogDrive `setup.py` `name="navsim" version="1.1.0"`，是 `autonomousvision/navsim` v1.1.0 的 fork，所以上游 NAVSIM 的 Python 约束完全传染过来——但仍是同一条 `>=3.9`，没有更紧的上限。
  - **物理约束**：5090 (Blackwell, sm_120) 需要 CUDA 12.8；PyTorch 从 2.5 起 cu128 nightly **不再发 cp39 wheel**。py3.9 + Blackwell = 死路。
  - **反向实证**：ReCogDrive issue #10 主楼 H100 用户就是 py3.10.0 跑通 ReCogDrive，最终与作者对齐到 PDMS=0.901，作者本人确认对齐。<https://github.com/xiaomi-research/recogdrive/issues/10>
  - **句法兼容性扫描**：Iniesta 扫 `nuplan-devkit v1.2` 874 个 .py 文件——0 处 PEP 604 union pipe、0 处 `match`/`case`、0 处 `@dataclass(slots=)`、0 处 `from collections import MutableMapping/...`、0 处 `np.<scalar>` 真代码调用、0 处 shebang 钉 `python3.9`；20+ 个文件用 `from __future__ import annotations`（PEP 563）做前向兼容。结论：py3.10 上 nuplan-devkit v1.2 不需要预备 monkey-patch。
- **Decision**: D5 = **Python = 3.10**。偏离上游 `environment.yml` 的软锁定，因为物理约束（cu128 + cp39 无 wheel）凌驾于 conda env 文件。
- **Consequence**:
  - Xavi 的 `env/setup_env.sh` 创建 `conda create -n navsim-rd python=3.10`，**不**沿用上游 `environment.yml`。
  - 残余风险：Shapely 2.0 API 变化（nuplan-devkit v1.2 有 41 处 `.coords / .geoms` 使用）。**该风险与 py 版本无关，独立 frozen**：上游 `requirements.txt` 钉 `Shapely>=2.0.0` 与 nuplan-devkit v1.2 自洽，我们不主动升 Shapely 就不踩；Xavi smoke test 真炸再扫。
  - `docs/dep_pinning_notes.md` 顶部需要落一条决策脚注（Xavi 装完顺手做）。

---

## Open / Pending decisions（待拍）

- **D2-followup**: v2 leaderboard 上线时 EC loss 配比方案 —— Iniesta **周五前**出草稿，Pep 拍。
- **D6 (potential)**: 复现达成 PDMS ≈ 86 后是否加 DriveVLA-W0 作对照 baseline —— Plan B 在 `docs/baseline_candidates.md` Addendum B 已 ready，待触发。

---

## 维护规则

1. 一条决策一段，不嵌套；想推翻就**新增 D-N**并把旧的标 `Superseded by D-N`，不要原地改。
2. Context 写"为什么要拍"，Decision 写"拍了什么"，Consequence 写"现在谁应该做什么 / 之前怎么做的现在不做了"。
3. 引用证据时贴**完整 URL**（GitHub blob link 优先），不要只写 issue 号或文件名。
4. `Frozen` 状态必须写"解冻条件"，否则就是 `Active`。
