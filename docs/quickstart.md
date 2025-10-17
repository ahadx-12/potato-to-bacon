# Quickstart

```bash
# run the API locally
uvicorn potatobacon.api.app:app --host 0.0.0.0 --port 8000 --reload

# simplest form: assignment
curl -s -X POST http://localhost:8000/v1/translate \
  -H "content-type: application/json" \
  -d '{"dsl":"E = m*c^2"}' | jq .

# still supported:
ptb translate examples/01_ke.dsl
ptb validate  examples/03_wave.dsl --domain classical
ptb codegen   examples/02_newton.dsl --name residual
ptb manifest  examples/01_ke.dsl --domain classical
```
