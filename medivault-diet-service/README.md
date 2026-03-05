# MediVault Diet Service

AI-powered personalised **Indian-cuisine** diet plan generator for the **MediVault Hospital Management System**.

Generates clinically-aware 7-day diet plans using patient medical records (diagnoses, medications, lab results, vitals, allergies) with automatic multi-provider LLM fallback.

---

## Features

- **Clinically-aware plans** — handles T2DM, HTN, CKD, hyperlipidemia, and 50+ other conditions
- **Drug-food interaction safety** — detects Metformin, Amlodipine, Atorvastatin, Warfarin, and more
- **Indian cuisine first** — full support for 7 regional cuisines (North Indian, South Indian, Gujarati, Maharashtrian, Bengali, Punjabi, Kerala)
- **Allergy enforcement** — hard blocks on shellfish, peanuts, dairy, gluten, etc.
- **Atlas Vector Search (RAG)** — retrieves relevant patient clinical context from ingested documents
- **Multi-provider LLM chain** — Groq → Gemini → Claude → OpenAI with automatic failover
- **Rate limiting** — 30 req/min global, 15 chat/min per patient
- **Security headers** — X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Cache-Control
- **Request tracing** — every request and log line tagged with `X-Request-ID`

---

## Architecture

```
┌──────────────┐     JWT      ┌──────────────────────────────────┐
│  Flutter App  │────────────►│  medivault-diet-service :5001     │
│  (Patient)    │◄────────────│  FastAPI + Uvicorn                 │
└──────────────┘   JSON       └──────┬─────────────┬──────────────┘
                                     │             │
                              ┌──────▼──┐    ┌─────▼──────────────┐
                              │ MongoDB  │    │  LLM Providers      │
                              │  Atlas   │    │  Groq → Gemini      │
                              │(diet DB) │    │  → Claude → OpenAI  │
                              └────┬─────┘    └────────────────────┘
                                   │
                        ┌──────────▼────────────┐
                        │ Atlas Vector Search    │
                        │ medical_vectors (RAG)  │
                        │ BAAI/bge-base-en-v1.5  │
                        └────────────────────────┘
                                   │
                        ┌──────────▼────────────┐
                        │ MediVault Backend API  │
                        │ (patient records,      │
                        │  diagnoses, meds, labs)│
                        └────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI 0.115+ |
| ASGI Server | Uvicorn + Gunicorn |
| Database | MongoDB Atlas (Motor + Beanie) |
| Auth | JWT (python-jose), shared secret |
| LLM Providers | Groq, Gemini, Claude, OpenAI |
| Embeddings | BAAI/bge-base-en-v1.5 (768 dims, local) |
| Rate Limiting | slowapi |
| Logging | structlog (JSON in production) |
| Validation | Pydantic v2 |
| Testing | pytest + pytest-asyncio |

---

## Quick Start

### 1. Clone and install

```bash
cd medivault-diet-service
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in MONGODB_URI, JWT_ACCESS_SECRET, and at least one LLM API key
```

### 3. Run

```bash
python server.py
```

Health check: `GET http://localhost:5001/api/v1/diet/health`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGODB_URI` | **Yes** | — | Atlas connection string |
| `JWT_ACCESS_SECRET` | **Yes** | — | Must match main MediVault backend |
| `GROQ_API_KEY` | One LLM key required | `""` | Groq API key |
| `GEMINI_API_KEY` | One LLM key required | `""` | Google AI Studio key |
| `ANTHROPIC_API_KEY` | One LLM key required | `""` | Anthropic key |
| `OPENAI_API_KEY` | One LLM key required | `""` | OpenAI key |
| `MEDIVAULT_API_BASE_URL` | No | `""` | Main backend URL for patient data |
| `MONGODB_DB_NAME` | No | `medivault_diet` | Diet service database name |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Active Groq model |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Active Gemini model |
| `LLM_FALLBACK_ORDER` | No | `groq,gemini,claude,openai` | Provider chain order |
| `CORS_ALLOWED_ORIGINS` | No | `http://localhost:3000,...` | Comma-separated origins |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `30` | Global rate limit |
| `RATE_LIMIT_CHAT_PER_MINUTE` | No | `15` | Chat rate limit |
| `NODE_ENV` | No | `development` | Set `production` to enable JSON logs and disable docs UI |

---

## API Endpoints

All endpoints are prefixed with `/api/v1/diet`. Protected endpoints require `Authorization: Bearer <JWT>`.

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Service health + provider status |
| `GET` | `/providers` | Patient JWT | List active LLM providers |
| `POST` | `/sessions` | Patient JWT | Create session and generate 7-day Indian diet plan |
| `GET` | `/sessions` | Patient JWT | List all sessions for the authenticated patient |
| `GET` | `/sessions/{id}` | Patient JWT | Get session detail and full message history |
| `POST` | `/sessions/{id}/chat` | Patient JWT | Send follow-up message (refine, explain, substitute) |
| `DELETE` | `/sessions/{id}` | Patient JWT | Soft-delete a session |

### POST /sessions — request body

```json
{
  "title": "My Diet Plan",
  "cuisine_preference": "indian",
  "regional_preference": "north_indian",
  "meal_count_preference": 5
}
```

`regional_preference` options: `north_indian`, `south_indian`, `gujarati`, `maharashtrian`, `bengali`, `punjabi`, `kerala`

---

## Indian Cuisine Configuration

The system prompt is tuned for Indian cuisine. Patients can specify a regional style at session creation or switch mid-conversation ("give me a South Indian version of this plan").

| `regional_preference` | Cuisine Style |
|---|---|
| `north_indian` | Roti, dal, sabzi, paneer-based |
| `south_indian` | Rice, sambar, rasam, idli, dosa |
| `gujarati` | Low-spice, lightly sweet, dhokla, thepla |
| `maharashtrian` | Bhakri, solkadhi, misal |
| `bengali` | Rice, fish (safe species), mustard-based |
| `punjabi` | Hearty dal, whole-wheat roti, low-fat lassi |
| `kerala` | Rice, coconut-based curries, fish |

---

## LLM Provider Setup

At least one provider key is required. The service skips providers with empty keys and falls back automatically on quota exhaustion.

### Recommended: Groq (free tier, fast)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Set `GROQ_API_KEY=gsk_...` and `GROQ_MODEL=llama-3.3-70b-versatile`

### Gemini (fallback)
1. Get a key from [aistudio.google.com](https://aistudio.google.com)
2. Set `GEMINI_API_KEY=AIza...` and `GEMINI_MODEL=gemini-2.0-flash`

---

## Vector Store Setup

An Atlas Vector Search index named `vector_index` with `numDimensions: 768` is required in the `medivault.medical_vectors` collection.

To ingest patient documents:

```bash
python scripts/ingest_patient_to_vectors.py
```

---

## Project Structure

```
medivault-diet-service/
├── server.py                  # Dev entry point (Uvicorn)
├── Procfile                   # Render.com web process
├── ecosystem.config.js        # PM2 process config
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Test dependencies
├── .env.example               # Environment variable template
├── .gitignore
├── FINAL_TEST_REPORT.md       # E2E test results (46/47, 97.9%)
├── app/
│   ├── main.py                # FastAPI factory + lifespan
│   ├── config/
│   │   ├── settings.py        # Pydantic Settings (all from .env)
│   │   └── database.py        # MongoDB + Beanie init
│   ├── models/
│   │   ├── diet_session.py    # Session document
│   │   └── diet_message.py    # Message document
│   ├── schemas/
│   │   ├── diet.py            # Request / Response schemas
│   │   └── patient.py         # PatientContext schemas
│   ├── services/
│   │   ├── llm/
│   │   │   ├── base.py        # Abstract LLM + exceptions
│   │   │   ├── groq_adapter.py
│   │   │   ├── gemini_adapter.py
│   │   │   ├── claude_adapter.py
│   │   │   ├── openai_adapter.py
│   │   │   └── router.py      # Multi-provider fallback chain
│   │   ├── patient_context.py # Fetch patient data (API + MongoDB fallback)
│   │   ├── diet_session.py    # Session CRUD
│   │   ├── diet_chat.py       # Orchestration + cuisine switching
│   │   └── vector_context.py  # Atlas Vector Search (RAG)
│   ├── routers/
│   │   ├── diet.py            # Diet endpoints
│   │   └── health.py          # Health + provider status
│   ├── middleware/
│   │   ├── auth.py            # JWT validation
│   │   ├── request_id.py      # X-Request-ID injection
│   │   ├── security_headers.py # OWASP security headers
│   │   └── error_handler.py   # Global exception handlers
│   └── utils/
│       ├── logger.py          # structlog setup
│       ├── prompt_builder.py  # Indian cuisine system prompt builder
│       ├── provider_state.py  # Quota exhaustion cooldown tracker
│       └── response_parser.py # Diet plan Markdown parser
├── scripts/
│   ├── gen_jwt.py             # Generate a single patient JWT
│   ├── _gen_jwts.py           # Generate test JWTs from .env
│   ├── _start.py              # Subprocess launcher for server
│   ├── _e2e_runner.py         # Full E2E test runner (47 tests)
│   └── ingest_patient_to_vectors.py  # Atlas Vector ingest
└── tests/
    ├── conftest.py            # Shared fixtures + TEST_JWT_SECRET
    ├── test_llm_router.py     # Router + fallback tests
    ├── test_diet_session.py   # Session CRUD tests
    ├── test_diet_chat.py      # Chat + prompt tests
    └── test_e2e.py            # Full integration test suite
```

---

## Deployment

### Render.com

Included `Procfile` and `render.yaml` handle configuration automatically. Set all environment variables in Render's dashboard.

### PM2

```bash
npm install -g pm2
pm2 start ecosystem.config.js
```

### Docker

```bash
docker build -t medivault-diet-service .
docker run -p 5001:5001 --env-file .env medivault-diet-service
```

---

## Testing

### Unit + integration (mocked, no external APIs)

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v --tb=short
```

68 tests, ~30s runtime.

### E2E (requires live server and LLM API quota)

```bash
python scripts/_start.py  # starts server
python scripts/_e2e_runner.py  # runs 47 tests
```

**Best recorded result: 46/47 PASS (97.9%)** — see `FINAL_TEST_REPORT.md`.

---

## Security

- JWT validated on every protected endpoint (HS256)
- Patients access only their own sessions (patient-ID-scoped queries)
- Input sanitised against XSS on session creation
- Rate limiting per IP and per patient
- Security response headers on all endpoints
- No hardcoded secrets — all config via `.env`
- Swagger/ReDoc UI disabled in production

---

## License

Part of the MediVault Hospital Management System.

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| Framework       | FastAPI 0.115+                    |
| ASGI Server     | Uvicorn + Gunicorn                |
| Database        | MongoDB Atlas (Motor + Beanie)    |
| Auth            | JWT (python-jose), shared secret  |
| LLM Providers   | Groq, Gemini, Claude, OpenAI      |
| Rate Limiting   | slowapi                           |
| Logging         | structlog (JSON)                  |
| Validation      | Pydantic v2                       |
| Testing         | pytest + pytest-asyncio           |

## Quick Start

### 1. Clone & Install

```bash
cd d:\MediVault\Microservices\medivault-diet-service
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your actual keys
```

Required environment variables:
- `MONGODB_URI` — MongoDB Atlas connection string
- `JWT_SECRET` — Same secret as your Node.js MediVault backend
- `MEDIVAULT_API_URL` — URL to the main MediVault API (e.g. `http://localhost:4000/api`)
- At least ONE LLM API key: `GROQ_API_KEY`, `GEMINI_API_KEY`, `CLAUDE_API_KEY`, or `OPENAI_API_KEY`

### 3. Run

```bash
# Development
python server.py

# Production
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5001 -w 2
```

### 4. Test

```bash
pip install -r requirements-dev.txt
pytest -v
```

## API Endpoints

All endpoints are prefixed with `/api/v1/diet`.

| Method   | Path                              | Auth | Description                          |
|----------|-----------------------------------|------|--------------------------------------|
| `GET`    | `/health`                         | No   | Health check                         |
| `GET`    | `/providers`                      | Yes  | LLM provider status                  |
| `POST`   | `/sessions`                       | Yes  | Create session & generate diet plan  |
| `POST`   | `/sessions/{id}/messages`         | Yes  | Send chat message to refine plan     |
| `GET`    | `/sessions/{id}`                  | Yes  | Get session details + current plan   |
| `GET`    | `/sessions/{id}/messages`         | Yes  | Get full chat history                |
| `GET`    | `/sessions`                       | Yes  | List patient's sessions              |
| `PATCH`  | `/sessions/{id}/complete`         | Yes  | Mark session as completed            |
| `DELETE` | `/sessions/{id}`                  | Yes  | Delete session and messages          |

## LLM Provider Fallback

The service tries providers in priority order. If a provider hits rate limits or quota exhaustion, it automatically falls back to the next one — **without losing conversation context**.

**Default priority:** Groq → Gemini → Claude → OpenAI

Configure via `LLM_FALLBACK_ORDER` env var (comma-separated).

## Project Structure

```
medivault-diet-service/
├── server.py                  # Dev entry point
├── Procfile                   # Render.com deployment
├── ecosystem.config.js        # PM2 deployment
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Test dependencies
├── .env.example               # Environment template
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app factory + lifespan
│   ├── config/
│   │   ├── settings.py        # Pydantic Settings (env vars)
│   │   └── database.py        # MongoDB + Beanie init
│   ├── models/
│   │   ├── diet_session.py    # Beanie Document
│   │   └── diet_message.py    # Beanie Document
│   ├── schemas/
│   │   ├── patient.py         # PatientContext models
│   │   └── diet.py            # Request/Response schemas
│   ├── services/
│   │   ├── llm/
│   │   │   ├── base.py        # ABC + custom exceptions
│   │   │   ├── groq_adapter.py
│   │   │   ├── gemini_adapter.py
│   │   │   ├── claude_adapter.py
│   │   │   ├── openai_adapter.py
│   │   │   └── router.py      # Fallback router
│   │   ├── patient_context.py # Fetch from MediVault API
│   │   ├── diet_session.py    # Session CRUD
│   │   └── diet_chat.py       # Orchestration layer
│   ├── routers/
│   │   ├── diet.py            # All diet endpoints
│   │   └── health.py          # Health + provider status
│   ├── middleware/
│   │   ├── auth.py            # JWT validation
│   │   └── error_handler.py   # Global exception handlers
│   └── utils/
│       ├── logger.py          # structlog configuration
│       ├── prompt_builder.py  # System prompt template
│       ├── provider_state.py  # Exhaustion cooldown tracker
│       └── response_parser.py # Diet plan Markdown parser
└── tests/
    ├── conftest.py            # Shared fixtures
    ├── test_llm_router.py     # Router + fallback tests
    ├── test_diet_session.py   # Session logic tests
    └── test_diet_chat.py      # Chat + prompt + parser tests
```

## Deployment

### Render.com

Uses the included `Procfile`:
```
web: gunicorn app.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT -w 2 --timeout 120
```

### PM2

```bash
pm2 start ecosystem.config.js
```

## License

Part of the MediVault Hospital Management System.
