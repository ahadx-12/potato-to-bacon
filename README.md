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

## Units API

| Endpoint | Description |
| --- | --- |
| `POST /v1/units/parse` | Parse multiline `name: unit` text into a map and return any warnings. |
| `POST /v1/units/validate` | Validate a units map, returning canonical SI strings and diagnostics. |
| `POST /v1/units/infer` | Infer missing units from a DSL expression and known units, returning a reasoning trace. |
| `POST /v1/units/suggest` | Suggest units for detected variables using SI/CGS/Imperial/Natural presets. |

`/v1/translate` and `/v1/validate` now embed normalized unit maps and dimensional explanations within their `report` payloads.

## Deploy
- Local: `make run`
- Docker: `docker build . && docker run -p 8000:8000 -e PTB_DATA_ROOT=/data potatobacon:dev`
- Railway: `railway.json` describes the start command; set `PTB_DATA_ROOT=/data`
