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

## Install log — 2026-05-14 (Xavi, T0 close-out)

`env/py310-navsim-cu128/` is the canonical training/inference venv. Built on this host (`ecs-5090-cb2-0007`, 8× RTX 5090 sm_120).

### What's actually installed (`pip list`, top of stack)

- `torch==2.12.0.dev20260407+cu128` — nightly, **must not be overwritten**. Installed via `pip install --no-deps /root/.cache/wheels-cu128/torch-*.whl` from a pre-fetched aria2 wheel.
- `cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1` — this is the umbrella metapackage that pulls in cudart/cublas/etc. Torch's `Requires-Dist` lists it explicitly; **don't skip it**, otherwise `import torch` dies with `OSError: libcudart.so.12: cannot open shared object file`.
- `cuda-bindings==12.9.6` — torch nightly requires `>=12.9.4,<13`.
- `nvidia-cudnn-cu12==9.20.0.48` / `nvidia-cusparselt-cu12==0.7.1` / `nvidia-nccl-cu12==2.29.7` / `nvidia-nvshmem-cu12==3.4.5` — torch's other direct cu12 reqs.
- `triton` — **not yet installed**. Torch wants `triton==3.7.0+git9c288bc5` (a PyTorch local-version tag) which is **not on tuna / not on PyPI**. Public tuna has `triton==3.7.0+git282c8251` (different commit hash). Smoke + matmul + cuDNN + NCCL all work without triton; only `torch.compile` paths will break. When Messi needs `torch.compile`, source the PyTorch-hosted triton wheel from `https://download.pytorch.org/whl/nightly/` (no R2 copy) and `pip install --no-deps` it.
- `numpy==2.2.6` — installed to clear the lazy `Failed to initialize NumPy` warning on `import torch`. Capped `<2.3` to leave headroom for downstream stacks. **NB**: this will be downgraded by Messi to `numpy==1.26.4` per ReCogDrive pin (see Other-pins table above). The post-downgrade state is the canonical one.
- Plain-python deps: `filelock 3.29.0`, `typing-extensions 4.15.0`, `sympy 1.14.0` (+ `mpmath 1.3.0`), `networkx 3.4.2`, `jinja2 3.1.6` (+ `MarkupSafe 3.0.2`), `fsspec 2026.4.0`.

### 🚨 Machine-level gotcha — proxy / pip

This host has a global proxy env baked in:

```
http_proxy  = http://127.0.0.1:1080
https_proxy = http://127.0.0.1:1080
all_proxy   = http://127.0.0.1:1080
```

pip (and curl, and aria2 unless explicitly stripped) inherit this and **route every PyPI/tuna request through the local socks**. Observed effects:

- Throughput collapses to **~200-300 KB/s** even on tuna mirrors that are LAN-fast directly.
- Long-running ranged downloads hit `urllib3.exceptions.ProtocolError: Connection broken: IncompleteRead(...)` mid-flight — socks splits a keep-alive at a chunk boundary, pip gives up. Reproduced on `nvidia-nccl-cu12` (287 MB) at ~65 MB / 23%.
- Effect compounds because pip retries the *same* socks lane → never recovers.

**Fix — strip the proxy before any pip / wheel-fetch**:

```sh
# one-shot (preferred — does not leak into the shell):
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    env/py310-navsim-cu128/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...

# or per-session:
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

Speed after stripping: **11-14 MiB/s** on tuna direct (≈ 50-70× faster). `env/setup_env.sh` already exports the data paths but **does not** strip the proxy — downstream agents must do it themselves until we decide whether to bake it in.

**Why we don't bake `unset *_proxy` into `setup_env.sh` yet**: some `huggingface_hub` / git clone / external API calls *do* want the proxy (e.g. `download.pytorch.org`, raw GitHub LFS). One-shot stripping per pip call is the safer default. Revisit if downstream agents hit this twice.

### Download strategy that worked

1. Large CUDA wheels (`cudnn` 628 MB, `nccl` 281 MB, `cusparselt` 274 MB, `nvshmem` 133 MB, `torch` 794 MB) → `aria2c --max-connection-per-server=16 --split=16` against the tuna `/packages/<hash>/<wheel>` direct URL, with proxy stripped. Store under `/root/.cache/wheels-cu128/`.
2. `pip install --no-deps <local-wheel-paths>` to land them in the venv without letting pip re-resolve the world (otherwise pip will happily yank a stable `torch==2.12.0` into the resolution set and overwrite the nightly).
3. Small/pure-python deps → normal `pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple ...` with proxy stripped; tuna direct is fast enough.
4. Run `pip install ... cuda-toolkit[...] cuda-bindings` **with** deps (no `--no-deps`) so the cu12 metapackage fans out correctly. Safe here because torch is already installed at a higher version-locked pin and won't be re-resolved into a stable version.

### R2 / official source notes

- `https://download-r2.pytorch.org/whl/nightly/cu128/` hosts `torch-*.whl` and `torchvision-*.whl`. **Does not** mirror the `nvidia-*-cu12` family (404).
- For triton-with-PyTorch-pin, use `https://download.pytorch.org/whl/nightly/` (not R2) when needed.
- Cloudflare HEAD on the R2 torch wheel reported `content-length: 832367434`, which led to a false "95% stuck at 794 MB" reading earlier. The 832 MB number *is* the true file size; aria2 `--continue=true` finishes it cleanly when re-invoked.

## Notes for downstream agents

- **Messi (training)**: torch is nightly — any model code that does
  `torch.__version__ == "2.0.1"` will fail. Use `>=` checks or skip the assert.
  Before pip-installing the ReCogDrive deps, **strip proxy env vars** (see
  gotcha section above). Numpy will get pinned down from 2.2.6 to 1.26.4 per
  ReCogDrive — that's expected.
- **Suarez (eval)**: same proxy-strip rule applies when installing eval-side deps.
- **Neymar (data)**: unrelated to torch, but: env vars in `env/setup_env.sh`
  point at `/home/work/hanjianhua/data/*` — keep your download targets in sync
  with what's documented there. **Also**: for big dataset downloads from
  outside-CN sources (HuggingFace, OpenScene S3), keep the proxy *on* — that's
  what it's there for. Only strip it for pip/tuna fetches.
- **All**: if you see `~200 KB/s` from any pip/curl, suspect the proxy first;
  retry with `env -u *_proxy ...` before assuming the mirror is slow.
