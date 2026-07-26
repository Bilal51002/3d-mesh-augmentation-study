# DGNet Docker Environment

## Architecture

- Model: DGNet
- Framework: Jittor
- Execution mode: CPU
- Official repository: https://github.com/li-xl/DGNet
- Official commit: see `docs/dgnet-official-commit.txt`

## Build the image

```bash
docker compose build dgnet
```

## Open the DGNet container

```bash
docker compose run --rm dgnet
```

## Test Jittor

```bash
docker compose run --rm dgnet \
  python3.7 -m jittor.test.test_example
```

## Verify DGNet Python files

```bash
docker compose run --rm dgnet \
  python3.7 -m compileall -q jmesh tools
```

## Mounted directories

- `models/dgnet` → `/workspace/dgnet`
- `datasets` → `/workspace/datasets`
- `configs/dgnet` → `/workspace/configs`
- `results/dgnet` → `/workspace/results`
