# Riyadh Air Quality Intelligence Platform

An end-to-end MLOps system that predicts whether the **next hour** in Riyadh will
exceed a high PM2.5 pollution threshold. Data is collected from Open-Meteo,
transformed into a leakage-free feature table, used to train and compare three
models tracked in MLflow, served through a typed FastAPI endpoint, consumed by a
Streamlit frontend, monitored with Evidently, containerised with Docker Compose,
and deployed through Dokploy. This project developed during the Tuwaiq Applied Artificial Intelligence Bootcamp.

| | |
|---|---|
| **Prediction target** | `high_pollution_next_hour` — 1 when next-hour PM2.5 > 100 µg/m³ |
| **Data** | 2,176 hourly observations (90 days, Riyadh 24.7136 N / 46.6753 E) |
| **Selected model** | Random Forest, decision threshold 0.35 |
| **Test F1 / Recall** | **0.940 / 0.955** (ROC-AUC 0.974) |
| **Stack** | UV, FastAPI, Streamlit, MLflow, Evidently, Docker Compose, Dokploy |

---

## 1. Architecture

```
                     Open-Meteo API
                           |
                           v
   src/air_quality/collect.py  -->  data/raw/air_quality.json   (immutable)
                           |
                           v
   src/air_quality/features.py -->  data/processed/model_table.parquet
                           |
                           v
   src/air_quality/train.py   -->  MLflow runs  +  models/model.joblib
                           |
                           v
         FastAPI service  <------  Streamlit frontend  (HTTP only)
                |
                +---- prediction logs ----> data/monitoring/predictions.csv
                |
                +---- Evidently report ---> reports/monitoring.html

   All services --> Docker Compose --> Dokploy
```

The frontend never loads `model.joblib`. It talks to FastAPI over HTTP and
degrades gracefully when the API is unreachable.

## 2. Repository structure

```
air-quality-mlops/
├── app/
│   ├── api/main.py            FastAPI service (/health, /model-info, /predict)
│   └── frontend/app.py        Streamlit UI, HTTP client only
├── src/air_quality/
│   ├── collect.py             Open-Meteo collector + response validation
│   ├── features.py            Target, time/lag/rolling features
│   ├── train.py               Chronological split, MLflow, model selection
│   └── monitoring.py          Evidently reference vs current report
├── tests/
│   ├── test_collect.py        Collector validation (network stubbed)
│   ├── test_features.py       Feature + leakage unit tests
│   ├── test_model.py          Split, pipeline, artifact contract
│   ├── test_api.py            API contract and acceptance criteria
│   └── test_integration.py    Full vertical slice, end to end
├── data/{raw,processed,monitoring}/
├── models/model.joblib
├── reports/{monitoring.html,evaluation.json}
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── README.md
```

## 3. Installation (UV)

UV is mandatory; all Python commands go through `uv run`.

```bash
# Install UV if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repository-url>
cd air-quality-mlops

uv sync                    # installs from uv.lock, creates .venv
uv run python --version    # verify
```

Copy the environment template and adjust if you are not using the defaults:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://api:8000` | Where Streamlit reaches FastAPI |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server |
| `MLFLOW_EXPERIMENT` | `riyadh-air-quality` | Experiment name |

## 4. Running locally

Run the pipeline in order. Steps 1–3 are reproducible from scratch.

```bash
# 1. Collect ~90 days of hourly data (validates days + duplicate timestamps)
uv run python -m air_quality.collect

# 2. Build the feature table and target
uv run python -m air_quality.features

# 3. Start MLflow, then train (separate terminals)
uv run mlflow server --host 0.0.0.0 --port 5000
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m air_quality.train

# 4. Serve the API
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Serve the frontend (separate terminal)
API_URL=http://127.0.0.1:8000 uv run streamlit run app/frontend/app.py --server.port 8501

# 6. After making some predictions, generate the drift report
uv run python -m air_quality.monitoring
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

## 5. Data collection

`collect.py` queries two Open-Meteo endpoints (air quality and weather archive)
for Riyadh, then writes both payloads verbatim under a `metadata` block
recording coordinates, date range, collection date, and row counts. The raw file
is never mutated by later stages.

Before saving, `validate_hourly()` enforces the specification's checkpoints:
at least 60 days of hourly rows, no duplicate timestamps, and equal-length
series across every variable.

## 6. Feature engineering and leakage prevention

Ten features, all computed from information available at prediction time:

| Group | Features |
|---|---|
| Current conditions | `pm2_5`, `pm10`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m` |
| Time | `hour`, `day_of_week` |
| History | `pm2_5_lag_1`, `pm2_5_lag_3`, `pm2_5_rolling_mean_6` |

Three safeguards, each covered by a test in `tests/test_features.py`:

1. **The target looks forward, the features never do.** The label is
   `pm2_5.shift(-1) > threshold`; no shifted-future column is kept as a feature.
2. **The rolling mean is shifted before the window.**
   `pm2_5.shift(1).rolling(6).mean()` excludes the current hour, so hour *t*'s
   own reading cannot leak into its own 6-hour average.
3. **Splits are chronological, never random.** 70% train / 15% validation /
   15% test by position, so the model is always tested on the future.

### Choosing the pollution threshold

The threshold defining a "high pollution" hour is a team decision. Riyadh's air
is dust-dominated — median PM2.5 is **122.9 µg/m³** — so the commonly used
35 µg/m³ cut-off labels **98.9%** of hours as positive. Under that target a
constant "always high risk" classifier scores F1 = 0.994, making the metric
useless.

We use **100 µg/m³**, which sits in the EPA "Unhealthy" band and is the only
candidate that keeps every chronological split non-degenerate:

| Threshold | Overall | Train | Valid | Test |
|---|---|---|---|---|
| 35 µg/m³ | 98.9% | — | — | (5 negatives in test) |
| 100 µg/m³ **(chosen)** | 65.2% | 66.6% | 69.0% | 54.7% |
| 125 µg/m³ | 48.9% | 53.0% | 46.3% | 32.6% |
| 150 µg/m³ | 34.5% | 39.1% | 31.4% | 16.2% |

## 7. Training, MLflow, and model selection

Every candidate is wrapped in the same pipeline — `SimpleImputer(median)` →
`StandardScaler` → classifier — with features selected **by name**, so an extra
column in the parquet can never silently reach the model.

Three runs are logged to MLflow per training invocation:

| Run | Validation F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| `baseline_most_frequent` | 0.8167 | 1.0000 | 0.6902 | 0.5000 |
| `logistic_regression` | 0.9526 | 0.9822 | 0.9247 | 0.9812 |
| `random_forest` **(selected)** | **0.9552** | 0.9467 | 0.9638 | 0.9809 |

**Model choice and the decision threshold are both made on the validation
slice.** Thresholds from 0.20 to 0.80 are scanned for best validation F1; the
winner was **0.35**. The test slice is read exactly once, after everything is
frozen, so the held-out numbers are honest.

### Model card

| | |
|---|---|
| **Model** | Random Forest (250 trees, max depth 10, `class_weight="balanced"`) |
| **Features** | The 10 listed above; median-imputed and standardised |
| **Target** | Next-hour PM2.5 > 100 µg/m³ |
| **Decision threshold** | 0.35, selected on validation data only |
| **Training data** | 1,523 hourly rows; validation 326; test 327 |
| **Test F1** | 0.9396 |
| **Test Recall (high risk)** | 0.9553 |
| **Test Precision** | 0.9243 |
| **Test ROC-AUC** | 0.9743 |

### Evaluation and error analysis

Test-set confusion matrix at threshold 0.35:

|  | Predicted normal | Predicted high |
|---|---|---|
| **Actually normal** | 134 (TN) | 14 (FP) |
| **Actually high** | 8 (FN) | 171 (TP) |

The threshold is deliberately below 0.50. For a public-health alert a missed
high-pollution hour (false negative) is worse than a false alarm, so the scan
settles where recall (0.955) exceeds precision (0.924) — 8 missed alerts versus
14 unnecessary ones.

Remaining errors cluster at **transitions**, where PM2.5 crosses 100 µg/m³
between consecutive hours and the lag features still describe the previous
regime. The base rate also drifts across the 90-day window (66.6% positive in
train, 54.7% in test), which is why test F1 lands slightly below validation F1 —
expected behaviour for a chronological split, and the reason the Evidently drift
report matters operationally.

Full numbers, including ROC curve points, are written to
`reports/evaluation.json` by the training script.

## 8. FastAPI service

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness and `model_loaded` flag |
| `/model-info` | GET | Model name, version, decision threshold, feature list |
| `/predict` | POST | Probability, binary prediction, and risk level |
| `/docs` | GET | Swagger UI |

The artifact is loaded **once at import**, never per request. Requests are
validated by a typed Pydantic schema; anything malformed returns 422 before
reaching the model.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"pm2_5":120,"pm10":260,"temperature_2m":34,"relative_humidity_2m":25,
       "wind_speed_10m":12,"hour":14,"day_of_week":2,"pm2_5_lag_1":118,
       "pm2_5_lag_3":110,"pm2_5_rolling_mean_6":115}'
```

```json
{"request_id":"054a65e4-...","prediction":1,"probability":0.7507,
 "risk_level":"high","model_version":"1.0.0"}
```

> **Note on validation bounds.** `pm10` accepts up to 5000 µg/m³ rather than the
> 1500 used in the course sample. Riyadh dust storms genuinely reach 3263 µg/m³
> in the collected data, and a 1500 ceiling rejected 26% of real observations
> with a 422. The bound is a guard rail against nonsense input, not a
> distribution limit.

Every prediction is appended to `data/monitoring/predictions.csv` with all ten
input features, the probability, the prediction, a request ID, and a UTC
timestamp. A logging failure is warned about but never costs the caller their
response.

## 9. Streamlit frontend

A single form collecting the ten features, grouped into air quality, weather,
time, and history. On submit it POSTs to `/predict` and renders the probability
and risk level.

The frontend holds **no model code**. It polls `/health` on load and shows
"FastAPI is unavailable" when the API is down, without falling back to any local
prediction path.

## 10. Monitoring with Evidently

```bash
uv run python -m air_quality.monitoring
```

- **Reference batch** — `data/monitoring/reference.csv`, the held-out test slice
  written by `train.py` (327 rows).
- **Current batch** — `data/monitoring/predictions.csv`, the live inputs logged
  by the API (150 rows in the committed report).

`DataSummaryPreset` and `DataDriftPreset` compare the two across all ten
features and write `reports/monitoring.html`. Because the reference batch is the
exact slice the model was scored on, any drift flagged in the report is a direct
signal that live traffic has moved away from the conditions the reported metrics
were measured under.

## 11. Docker and Docker Compose

```bash
docker compose config          # validate
docker compose up --build -d   # build and start
docker compose ps              # all services healthy
curl http://localhost:8000/health
docker compose logs -f api
docker compose down            # stop, keep volumes
```

Three services:

| Service | Port | Notes |
|---|---|---|
| `api` | 8000 | Health check, `./models` mounted read-only |
| `frontend` | 8501 | Starts only after `api` is healthy |
| `mlflow` | 5000 | SQLite backend store on a named volume |

Both `monitoring_data` and `mlflow_data` are named volumes, so prediction logs
and experiment history survive `docker compose down` and Dokploy redeploys. All
three services use `restart: unless-stopped`.

## 12. Dokploy deployment

1. Push the repository to a Git provider the Dokploy server can reach.
2. Create a new **Docker Compose** project in Dokploy and connect the repository
   and branch.
3. Set the Compose file path to `docker-compose.yml`. Enable automatic
   deployment only after the first manual deploy succeeds.
4. Add environment variables in the Dokploy UI (`API_URL`,
   `MLFLOW_TRACKING_URI`). Never commit `.env`.
5. Assign a domain to `frontend` on port 8501, and a separate domain or
   protected route to `api` on port 8000.
6. Deploy, watch the build logs, and wait for the API health check to pass.
7. Smoke test: open `/health` and `/docs` on the API domain, then submit one
   prediction from the Streamlit domain.
8. Redeploy and confirm MLflow history and prediction logs survived, proving the
   named volumes are persistent.

### Runbook

| Situation | Action |
|---|---|
| API returns 503 / unhealthy | `docker compose logs api`; confirm `models/model.joblib` is present and readable |
| Frontend shows "FastAPI is unavailable" | Check `API_URL`; inside Compose it must be `http://api:8000`, not `localhost` |
| MLflow experiment is empty | `MLFLOW_TRACKING_URI` was unset when training ran; re-run training with it set |
| Drift report will not build | Needs at least one logged prediction; POST to `/predict` first |
| Model needs retraining | Re-run collect → features → train, then redeploy so the new artifact is mounted |
| Roll back | Redeploy the previous commit in Dokploy; volumes are untouched |
| Full local reset | `docker compose down -v` (deletes volumes — instructors only) |

## 13. Testing

```bash
uv run ruff check .
uv run pytest
```

**61 tests, all passing.** No test requires network access or a running MLflow
server; the Open-Meteo response is stubbed and synthetic data is generated in
temporary directories.

| File | Covers |
|---|---|
| `test_collect.py` | Response validation, duplicate timestamps, minimum window, HTTP errors |
| `test_features.py` | Target alignment, lag correctness, rolling-window leakage, threshold behaviour |
| `test_model.py` | Split disjointness and ordering, pipeline composition, artifact contract, beating baseline |
| `test_api.py` | All three endpoints, 422 cases, threshold agreement, prediction logging, Swagger |
| `test_integration.py` | features → train → artifact → API → prediction log → Evidently report |

The leakage tests are the important ones:
`test_rolling_mean_excludes_the_current_hour` asserts the shifted window differs
from an unshifted one, so removing the `.shift(1)` fails the suite rather than
silently inflating scores.

### Acceptance tests

| Test | Command | Result |
|---|---|---|
| Environment | `uv sync && uv run pytest` | ✅ 61 passed |
| Data pipeline | `uv run python -m air_quality.{collect,features}` | ✅ 2,176 rows |
| Training | `uv run python -m air_quality.train` | ✅ artifact + metrics |
| API health | `GET /health` | ✅ 200, `model_loaded=true` |
| Input validation | `POST /predict` incomplete | ✅ 422 |
| Valid prediction | `POST /predict` complete | ✅ probability 0.7507 |
| Frontend separation | Stop the API | ✅ reports unavailable, no fallback |
| Containers | `docker compose up --build -d` | ✅ all healthy |
| Monitoring | Open `reports/monitoring.html` | ✅ 327 reference vs 150 current |
| Deployment | Open Dokploy domains | ✅ see deployment section |

## 14. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: air_quality` | Run through UV (`uv run python -m ...`); the package is installed from `src/` by `uv sync` |
| `FileNotFoundError: model_table.parquet` | Run `uv run python -m air_quality.features` before training |
| Training warns "MLflow unavailable" | The tracking server is not running. Training still completes and saves the artifact; start MLflow and re-run to log the experiment |
| `Untrusted types found in the file` from MLflow | Fixed by logging with `serialization_format="cloudpickle"`; the default skops backend rejects `numpy.dtype` inside a fitted `ColumnTransformer` |
| API fails at startup with `KeyError: 'pipeline'` | An old-format `model.joblib`. Re-run training to write the `{pipeline, threshold, ...}` artifact |
| `POST /predict` returns 422 on real data | Check the value against the schema bounds; PM10 above 5000 µg/m³ is rejected by design |
| Streamlit cannot reach the API in Docker | Use the service name `http://api:8000`; `localhost` inside a container refers to the container |
| Port already in use | `pkill -f uvicorn` or change the published port in `docker-compose.yml` |
| Evidently report is empty or errors | The prediction log needs the feature columns; delete a stale `predictions.csv` written by an older schema and re-predict |
| Drift results look meaningless | Fewer than ~5 current rows. Submit more predictions before generating the report |
