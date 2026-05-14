# navsim-auto-research

Working tree for our NAVSIM-based autonomous-driving research run.

## Layout

```
.
├── env/                          # Python venvs + env setup script
│   ├── py310-navsim-cu128/       # the venv (not tracked)
│   └── setup_env.sh              # `. env/setup_env.sh` to activate + export vars
├── third_party/                  # upstream repos (not tracked, cloned shallow)
│   ├── navsim/                   # autonomousvision/navsim — base devkit
│   └── recogdrive/               # xiaomi-research/recogdrive — our model recipe
├── scripts/                      # helper scripts (e.g. download_navsim_mini.sh)
├── docs/                         # design notes, entrypoint docs, version pin notes
└── .github/workflows/ci.yml      # lint + import smoke (NO training/eval in CI)
```

## Quickstart

```sh
. env/setup_env.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import navsim; print('navsim ok')"
```

## Owners

| area | owner |
|---|---|
| ops / env / inference serving | Xavi |
| training entrypoints | Messi |
| eval / leaderboard | Suarez |
| data download / dataset layout | Neymar |
| arch / decisions | Iniesta |
| supervisor | Pep |

## See also

- `docs/dep_pinning_notes.md` — every place we diverge from upstream pins
- `env/setup_env.sh` — canonical env vars (`NUPLAN_MAPS_ROOT`, `OPENSCENE_DATA_ROOT`, ...)
