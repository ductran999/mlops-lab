# mlops-lab

A dual-track learning lab:

- **AI Applied Engineer (primary):** build a data analyst agent
  (text-to-SQL + charting) with evals and observability.
- **AI Platform (secondary):** model serving, experiment tracking, model registry —
  KServe/ONNX labs and the MLflow + Postgres + MinIO stack.

The `app/` scaffold is intentionally **empty**: folders are created first, code is
filled in incrementally while studying.

## Structure (folders to fill in)

```text
app/                        # data analyst agent (production-style code)
├── main.py                 # TODO: FastAPI entrypoint
├── api/                    # TODO: routes (POST /chat, streaming)
├── core/                   # TODO: settings, env, logging
├── agent/                  # TODO: LLM client, tools (run_sql, make_chart), agent loop
├── evals/                  # TODO: golden set + mlflow.evaluate runner
│   └── datasets/
└── observability/          # TODO: MLflow tracing setup

labs/                       # learning labs — one subfolder per topic, experiments only
├── main.py                 # first Gemini SDK example
├── company.md              # shared document for RAG labs
└── rag_basic/              # lab: in-context learning vs basic RAG comparison

models/onnx/                # ONNX export + KServe serving lab
notebooks/                  # exploration notebooks (house price)
scripts/                    # infra utilities (check_infra.py)
data/eval/                  # seed data for agent evals (empty for now)
tests/                      # TODO: unit + integration tests

docker-compose.yaml         # MLflow + Postgres + MinIO
```

## Infra

```bash
docker compose up -d          # MLflow :5000, Postgres :5432, MinIO :9000/:9001
make healthcheck
```

## Dependencies

Managed with Poetry (`pyproject.toml`, `package-mode = false`):

```bash
poetry install
```
