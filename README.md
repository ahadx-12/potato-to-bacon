# potato.to.bacon

Physics-first translator (DSL → validated schema → guards → canonical form). MVP focuses on:
- DSL (typed params, equations, function calls)
- Deterministic canonicalization
- Pint-backed dimensional validation with YAML registries
- Schema builder + JSON schema
- Tests

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Deploy
- Local: `make run`
- Docker: `docker build . && docker run -p 8000:8000 -e PTB_DATA_ROOT=/data potatobacon:dev`
- Railway: `railway.json` describes the start command; set `PTB_DATA_ROOT=/data`
