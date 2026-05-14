# NAVSIM Baseline 选型短名单

**作者**: Iniesta (Researcher)
**日期**: 2026-05-14
**目标**: 选 1–2 个开源自动驾驶 VLA 作为我们在 NAVSIM 上的复现 baseline，要求在 **8× RTX 5090 (32 GB)** 上一周内能跑通 mini 评测（`navtest` 的小子集或 v2 `private_test_hard` warm-up）。
**扫描范围**: GitHub 近 12 个月（2025-05 ~ 2026-05），star ≥ 100 的端到端 / planning 类自动驾驶 VLA / VLM-planner，参考方法谱：DriveVLM、EMMA、LMDrive、DriveLM、Senna、Hydra-MDP、OpenEMMA、AutoVLA、ReCogDrive、DriveVLA-W0、FSDrive、SimLingo、OpenDriveVLA。

## TL;DR — 推荐排序

| Rank | Repo | 选它的理由 | 风险点 |
|---|---|---|---|
| **#1 首选** | `xiaomi-research/recogdrive` (ReCogDrive, ICLR'26) | 原生 NAVSIM 评测脚本齐 + 已开源 2B / 8B 全套权重 + 三阶段（VLM → IL → RL）都放出来了 + PDMS 90.8 是当下开源最强档；2B 模型推理 / LoRA 微调单卡 5090 32GB 妥妥够 | RL 阶段对显存最敏感，全量 8B + GRPO 单卡装不下；但我们 baseline 目标只要 Stage-1/2 复现就够 |
| **#2 备选** | `BraveGroup/DriveVLA-W0` (ICLR'26) | 直接给 NAVSIM v1.1 推理脚本 + 已开源 PDMS=87.2 checkpoint；目录里 `navsim/` 子树就是为 NAVSIM 准备的；官方训练用 8×L20-40GB ~16h，迁到 8×5090-32GB 训练略紧但 **推理 + LoRA 没问题** | 主干用 Emu3 + flow-matching action expert，链路重；从头训需要重写显存/精度配置 |
| #3 | `ucla-mobility/AutoVLA` (NeurIPS'25) | 把 navsim devkit 直接内嵌进 repo；Qwen2.5-VL-3B 主干，量级最友好；带 nuPlan / Waymo / CARLA / nuScenes 多基准 | CoT 标注用 Qwen2.5-VL-72B，AWQ 4bit 在 5090 上也得 2 张卡拼，预处理慢；数据集 "Coming Soon" 标 |
| #4 | `DriveVLA/OpenDriveVLA` (AAAI'26) | 0.5B 模型权重已开源，门槛最低，适合快速摸 pipeline | 主战场是 nuScenes，NAVSIM 评测要自己适配；mmcv / mmdet3d 老依赖坑多 |
| #5 | `MIV-XJTU/FSDrive` (NeurIPS'25 spotlight) | 视觉 CoT 想法新颖，能给我们方法迭代留口子 | nuScenes-only，没 NAVSIM 适配，作 baseline 不划算，留作 idea 借鉴 |

> 出局但提一下：`hustvl/Senna` (550★) 偏 meta-action + 端到端两段式，NAVSIM 没有现成入口；`opendilab/LMDrive` / `RenzKa/simlingo` 是 CARLA 闭环路线，与 NAVSIM (nuPlan-based, open-loop + pseudo-sim) 评测范式不匹配；`NVlabs/Hydra-MDP` 没公开训练代码，只有方法描述。

---

## 候选详表

### 1. ReCogDrive — `xiaomi-research/recogdrive` ⭐528

- **Repo**: <https://github.com/xiaomi-research/recogdrive>
- **Paper**: ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving, Xiaomi Research, ICLR 2026 (arXiv:2506.08052)
- **Stars / Forks**: 528 / 65
- **Last commit**: 2026-03-13 (活跃)
- **Repo created**: 2025-06-05
- **NAVSIM 适配**: ✅ **原生**。README 给了完整 NAVSIM 训练/评测脚本，引用 `autonomousvision/navsim` 官方安装；NAVSIM v1 `navtest` 上报 PDMS 90.8 (2B-RL) / 90.4 (8B-RL)。NAVSIM v2.0 评测框架 README 里标了 `[ ]`，是 TODO（我们要刷 v2 leaderboard 得自己接 v2 devkit，工作量 ≈ 1–2 人天）。
- **方法**: VLM (Qwen2-VL 系) → IL trajectory head (~35M) → GRPO RL，三阶段。
- **HF 模型**: 全开放 [owl10/ReCogDrive-*](https://huggingface.co/owl10)，**不 gated**。`HF_TOKEN` 配上就能拉。
- **训练数据**: 预训练 mix（NAVSIM-Traj / NAVSIM-ReCogDrive / DriveLM / NuInstruct / NuscenesQA / OmniDrive / Senna / LingoQA / Drama），作者打包成 JSONL 一并放在 HF。下游 NAVSIM v1 trainval ~120k frames。
- **单卡显存估算 (5090 32GB)**:
  - 2B-VLM 推理 fp16: ~6 GB；LoRA 微调 bs=1 ~14 GB；全参 bs=1 ~26 GB → 单卡可行
  - 8B-VLM 推理 fp16: ~20 GB；LoRA 微调勉强 26–30 GB；全参单卡装不下 → ZeRO-2 多卡
  - Stage-3 RL (GRPO，actor+ref) 2B 模型单卡可，8B 必须切分
- **跑通 mini 评测的最短路径** (符合一周窗口):
  1. clone repo + 装 NAVSIM v1 devkit + 下 mini-trainval（Neymar 在做）
  2. 拉 `owl10/ReCogDrive-2B-IL` checkpoint
  3. 直接跑 inference + PDMS 评测 → 复现 86.5 PDMS
  4. （可选）拉 `owl10/ReCogDrive-2B-RL` 跑 90.8 PDMS

### 2. DriveVLA-W0 — `BraveGroup/DriveVLA-W0` ⭐355

- **Repo**: <https://github.com/BraveGroup/DriveVLA-W0>
- **Paper**: DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving, ICLR 2026 (arXiv:2510.12796)
- **Stars / Forks**: 355 / 24
- **Last commit**: 2026-02-11
- **Repo created**: 2025-10-09 (新，活跃)
- **NAVSIM 适配**: ✅ **原生 NAVSIM v1.1**。目录里直接有 `normalizer_navsim_test/`、`normalizer_navsim_trainval/`、`inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh`、`scripts/tokenizer/extract_vq_emu3_navsim.sh`。NAVSIM v2 也得自己接。
- **方法**: Emu3 主干 + flow-matching action expert + world-model 自监督扩样，路线偏重。
- **HF 模型**: [liyingyan/DriveVLA-W0](https://huggingface.co/liyingyan/DriveVLA-W0)，**不 gated**。
- **训练数据**: NAVSIM v1.1 trainval（与官方 navsim 一致）。
- **单卡显存估算 (5090 32GB)**:
  - 官方训练配方 8×L20-40GB ~16h，对显存吃得比较紧。迁到 5090-32GB 时 **训练需要 ZeRO-2/3 + 梯度检查点**，bs=1 + grad-accum 可以装下
  - 推理 + LoRA 单卡可行
- **风险**: Emu3 视觉 tokenizer 链路重，调起来调试时间不少；做 baseline 主要为了多一条对比，不当首选。

### 3. AutoVLA — `ucla-mobility/AutoVLA` ⭐539

- **Repo**: <https://github.com/ucla-mobility/AutoVLA>
- **Paper**: AutoVLA: A Vision-Language-Action Model with Adaptive Reasoning and RFT, NeurIPS 2025
- **Stars / Forks**: 539 / 40
- **Last commit**: 2026-02-03
- **NAVSIM 适配**: ✅ **半原生**。repo 把 `autonomousvision/navsim` devkit 整个内嵌（`navsim/` 子目录），下载脚本和环境变量都用 NAVSIM 那套；不过主报告基准在 nuPlan / nuScenes / Waymo / CARLA，NAVSIM PDMS 数表少。
- **方法**: Qwen2.5-VL-3B 主干 + 物理动作 token + CoT + GRPO RFT。
- **HF 模型**: 主干用 `Qwen/Qwen2.5-VL-3B-Instruct`（**不 gated**）；CoT 标注阶段用 `Qwen2.5-VL-72B`（也不 gated 但量大）。
- **训练数据**: nuPlan + nuScenes + DriveLM 标注 (`v1_1_train_nus.json`)。数据集 "Coming Soon" 字样还在，但 nuPlan / nuScenes 本来就能下到。
- **单卡显存估算 (5090 32GB)**: 3B 主干训练 bs=1 LoRA ~16 GB，全参 ~28 GB；72B CoT 标注阶段需要 2× 5090 + AWQ-4bit。
- **风险**: 数据准备链长（CoT 标注 → 动作离散化 → SFT → RFT），一周窗口里跑通 mini 评测需要砍掉 72B CoT 那段，改用预生成 CoT。

### 4. OpenDriveVLA — `DriveVLA/OpenDriveVLA` ⭐727

- **Repo**: <https://github.com/DriveVLA/OpenDriveVLA>
- **Paper**: OpenDriveVLA, AAAI 2026
- **Stars / Forks**: 727 / 69
- **Last commit**: 2026-02-16
- **NAVSIM 适配**: ❌ 主战场 nuScenes，**没有 NAVSIM 评测脚本**。需要自己写一层适配（trajectory 输出 → NAVSIM Agent 接口 ~ 2–3 人天）。
- **方法**: LLaVA-NeXT + Qwen2.5 backbone，主打 0.5B 轻量模型已开源。
- **HF 模型**: `OpenDriveVLA/OpenDriveVLA-0.5B`，**不 gated**。
- **依赖**: mmcv + mmdet3d 老链，5090 (sm_120) 上预编译大概率坏，要么从源码编 mmcv，要么换 torch nightly。注意 5090 需 torch ≥ 2.5 / CUDA 12.8 才稳。
- **结论**: 模型轻、上手快是优点，但 NAVSIM 适配 + mmcv 链路两个坑加起来不划算，作 #4 候选。

### 5. FSDrive — `MIV-XJTU/FSDrive` ⭐724

- **Repo**: <https://github.com/MIV-XJTU/FSDrive>
- **Paper**: FutureSightDrive (FSDrive): Thinking Visually with Spatio-Temporal CoT, NeurIPS 2025 spotlight
- **NAVSIM 适配**: ❌ 纯 nuScenes，评测用 L2 + 碰撞率，**与 NAVSIM PDMS 体系不接**。
- **价值**: 它的"用视觉 token 自回归预测未来帧 + 轨迹"的思路，可以作为我们后期的 method 创新点（在 ReCogDrive baseline 上加 visual CoT 头）。
- **结论**: 不选作 baseline，列入 idea 池。

---

## 推荐方案（给 Pep）

**Plan A（推荐）**: 单 baseline = **ReCogDrive-2B**。

- 第 1–2 天：装 NAVSIM v1 + v2 devkit、拉 HF 2B-IL / 2B-RL checkpoint。
- 第 3–4 天：跑 `navtest` mini split inference，复现 PDMS ≈ 86 / 90。
- 第 5 天：把评测脚本接到 v2 `private_test_hard` warm-up leaderboard（Suarez 负责提交格式）。
- 第 6–7 天：在 2B 上跑一轮 LoRA 微调，确认我们自己训出来的模型与官方 ckpt PDMS 差距 < 1，证明 pipeline 跑通。

**Plan B（双 baseline，稳）**: ReCogDrive-2B + DriveVLA-W0-推理。后者只跑推理 + 复现 PDMS=87.2，不训练，对照成本极低，但能给我们一条独立验证 NAVSIM 评测 pipeline 的链路。

**关键依赖**:
- Neymar：NAVSIM v1.1 mini-trainval + `private_test_hard` 数据落到 `/home/work/hanjianhua/data/navsim/` （这块他在并行做）
- Suarez：NAVSIM v2 PDMS 评测脚本 + HF leaderboard 提交格式
- Messi：ReCogDrive 三阶段训练脚本调参（Stage-2 IL 是首选先复现的）
- Xavi：5090 上的 torch + flash-attn + transformers 基础环境（torch 2.5+ / CUDA 12.8）

## Open questions

1. **NAVSIM v1 还是 v2 leaderboard？** ReCogDrive 报的数是 v1 `navtest`；当下 HF 公开 leaderboard 是 v2 (`AGC2025/e2e-driving-warmup-iccv` / `private_test_hard`)。如果目标是"刷榜露脸"必须上 v2，那 ReCogDrive 还得自己迁；如果先验证 pipeline，从 v1 navtest 起步。 → 请 Pep 拍板。
2. **是否复现 RL 阶段？** GRPO 在 2B 上 5090 单卡可跑，但 reward shaping（PDMS 子项 + 碰撞 + 进度）调参经验我们没有，可能花掉一周里 2–3 天。建议第一周只到 Stage-2 IL，第二周再考虑 RL。
3. **5090 (sm_120) 软件栈**：vllm / flash-attn / xformers 在 sm_120 上的兼容性需要 Xavi 先趟一遍，这个不是 baseline 选型问题，但会影响时间预估。

---

## 引用

- ReCogDrive: arXiv:2506.08052 (Xiaomi Research, ICLR 2026)
- DriveVLA-W0: arXiv:2510.12796 (Brave Group, ICLR 2026)
- AutoVLA: NeurIPS 2025 (UCLA Mobility Lab)
- OpenDriveVLA: AAAI 2026
- FSDrive: NeurIPS 2025 spotlight (MIV-XJTU)
- NAVSIM: arXiv:2406.15349 (autonomousvision, NeurIPS 2024) + arXiv:2506.04218 (CoRL 2025, Pseudo-Sim)
- Senna: hustvl, arXiv 2024
- DriveLM: ECCV 2024 Oral
- Hydra-MDP: NVlabs, 2024

> 注：以上对训练显存 / 训练时间的估算是 Iniesta 的推断，**不是论文里给的数字**。论文里只有 DriveVLA-W0 明写了 8×L20-40GB ~16h；ReCogDrive / AutoVLA 都没给硬件耗时表。落地后请以 Messi 实测为准。
