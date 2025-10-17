# Quickstart

```bash
uvicorn potatobacon.api.app:app --reload
ptb translate examples/01_ke.dsl
ptb validate  examples/03_wave.dsl --domain classical
ptb codegen   examples/02_newton.dsl --name residual
ptb manifest  examples/01_ke.dsl --domain classical
```
