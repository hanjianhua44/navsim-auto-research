# Dependency Pinning Notes

> **Top-level decisions (locked by Pep)**
>
> - **Python = 3.10.** Upstream NAVSIM/ReCogDrive `environment.yml` pins
>   `python=3.9` (soft, conda lock), but `setup.py` only requires `>=3.9`. We
>   deviate because cu128 nightly has no cp39 wheel and Blackwell needs cu128.
>   - *Compat scan (Iniesta, nuplan-devkit v1.2)*: **0 hits across 874 files**
>     for 3.10-removed APIs (`collections.MutableMapping` etc.), `np.bool/int/
>     float/object` in real code, `@asyncio.coroutine`, py3.9 shebangs, etc.
>     20+ files already use `from __future__ import annotations`. 0 monkey-
>     patches needed up front.
>   - *Residual risk*: Shapely 2.0 API churn (41 `.coords[]` / `.geoms` sites in
>     nuplan-devkit) — **frozen** until `import navsim` smoke-test actually
>     fails. Not py-version related.
> - **torch nightly +cu128** (Iniesta D4 / Pep). Reason: only path that supports
>   sm_120 (Blackwell / RTX 5090). Stable torch tops at sm_90.
> - **transformers >=4.37.0,<4.38**, **numpy==1.26.4**, **flash-attn==2.7.0.post2**
>   (ReCogDrive upstream pin set, via Iniesta + Pep). Reason: PDMS 86.5
>   reproduction. transformers range chosen because upstream issue #10 says
>   `4.37.0` but `internvl_chat/pyproject.toml` says `4.37.2` — both verified.
>   flash-attn has no sm_120 wheel — will source-build if pip fails.

Tracks every place we **deliberately diverge** from upstream pins. Three columns
per row: upstream pin / what we installed / why. Keep this honest — anyone
reading this should be able to reproduce our env without surprises.

Upstream sources tracked:
- `third_party/navsim/environment.yml` + `third_party/navsim/requirements.txt`
- (later) ReCogDrive `third_party/recogdrive/requirements.txt`

---

## Python interpreter

| upstream | ours | reason |
|---|---|---|
| python=3.9 (navsim/environment.yml) | python=3.10.12 (system, Ubuntu 22.04) | PyTorch nightly cu128 (required for Blackwell sm_120 / RTX 5090) ships wheels only for cp310/311/312/313/314. cp39 is not published on the cu128 nightly index. Choosing 3.10 (closest to upstream's 3.9). |

## CUDA / PyTorch stack

| upstream | ours | reason |
|---|---|---|
| torch==2.0.1 | torch nightly +cu128 (cp310) | torch 2.0.1 ships max sm_90 (Hopper). RTX 5090 is sm_120 (Blackwell). Stable releases do not yet cover sm_120 cleanly; nightly +cu128 is the current officially-supported path. |
| torchvision==0.15.2 | torchvision nightly +cu128 (cp310) | Must match torch build / cuda runtime. |

## Other pins (filled in as we install)

| upstream | ours | reason |
|---|---|---|
| pytorch-lightning==2.2.1 | TBD | PL 2.2.1 may or may not be compatible with torch nightly; if not, bump minor version. |
| tensorboard==2.16.2 | TBD | usually OK across torch versions. |
| numpy==1.23.4 (navsim) | **1.26.4 (forced)** | ReCogDrive upstream issue #10: numpy 1.23.* has a `linalg.inv` numerical bug that drops Stage-2/3 PDMS ~5 points. Upstream author confirms 1.26.4 required. Per Iniesta scan + Pep D-call. |
| transformers (unspec.) | **>=4.37.0,<4.38 (forced)** | ReCogDrive issue #10 author wrote `4.37.0`; their `internvl_chat/pyproject.toml` pins `4.37.2`. Both verified — range accepts either, hard-pin a single point is more brittle. Per Iniesta + Pep. |
| flash-attn (unspec.) | **2.7.0.post2 (forced) ⚠️** | ReCogDrive recommended pin. **No official pre-built wheel for sm_120 (Blackwell)** in flash-attn release matrix. Plan: try `pip install` first; fall back to building from source with `TORCH_CUDA_ARCH_LIST="12.0"` (CUDA 12.8 + ninja, ~30-45min). If source build fails, escalate to Pep. |
| setuptools==65.5.1 | 65.5.1 | matches upstream pin (matters for legacy `setup.py` builds). |
| nuplan-devkit @ git+...v1.2 | same | hard pin from upstream; keep. |

---

## Notes for downstream agents

- **Messi (training)**: torch is nightly — any model code that does
  `torch.__version__ == "2.0.1"` will fail. Use `>=` checks or skip the assert.
- **Suarez (eval)**: same.
- **Neymar (data)**: unrelated to torch, but: env vars in `env/setup_env.sh`
  point at `/home/work/hanjianhua/data/*` — keep your download targets in sync
  with what's documented there.
