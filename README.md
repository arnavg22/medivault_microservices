# MediVault Diet Service

AI-powered personalised **Indian-cuisine** diet plan generator for the **MediVault Hospital Management System**.

Generates clinically-aware 7-day diet plans using patient medical records (diagnoses, medications, lab results, vitals, allergies) with automatic multi-provider LLM fallback.

---

## Features

- **Clinically-aware plans** â€” handles T2DM, HTN, CKD, hyperlipidemia, and 50+ other conditions
- **Drug-food interaction safety** â€” detects Metformin, Amlodipine, Atorvastatin, Warfarin, and more
- **Indian cuisine first** â€” full support for 7 regional cuisines (North Indian, South Indian, Gujarati, Maharashtrian, Bengali, Punjabi, Kerala)
- **Allergy enforcement** â€” hard blocks on shellfish, peanuts, dairy, gluten, etc.
- **Atlas Vector Search (RAG)** â€” retrieves relevant patient clinical context from ingested documents
- **Multi-provider LLM chain** â€” Groq â†’ Gemini â†’ Claude â†’ OpenAI with automatic failover
- **Rate limiting** â€” 30 req/min global, 15 chat/min per patient
- **Security headers** â€” X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Cache-Control
- **Request tracing** â€” every request and log line tagged with `X-Request-ID`

---

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     JWT      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Flutter App  â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–ºâ”‚  medivault-diet-service :5001     â”‚
â”‚  (Patient)    â”‚â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚  FastAPI + Uvicorn                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   JSON       â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                     â”‚             â”‚
                              â”Œâ”€â”€â”€â”€â”€â”€â–¼â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                              â”‚ MongoDB  â”‚    â”‚  LLM Providers      â”‚
                              â”‚  Atlas   â”‚    â”‚  Groq â†’ Gemini      â”‚
                              â”‚(diet DB) â”‚    â”‚  â†’ Claude â†’ OpenAI  â”‚
                              â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                   â”‚
                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                        â”‚ Atlas Vector Search    â”‚
                        â”‚ medical_vectors (RAG)  â”‚
                        â”‚ BAAI/bge-base-en-v1.5  â”‚
                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                   â”‚
                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                        â”‚ MediVault Backend API  â”‚
                        â”‚ (patient records,      â”‚
                        â”‚  diagnoses, meds, labs)â”‚
                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
| `MONGODB_URI` | **Yes** | â€” | Atlas connection string |
| `JWT_ACCESS_SECRET` | **Yes** | â€” | Must match main MediVault backend |
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

### POST /sessions â€” request body

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
â”œâ”€â”€ server.py                  # Dev entry point (Uvicorn)
â”œâ”€â”€ Procfile                   # Render.com web process
â”œâ”€â”€ ecosystem.config.js        # PM2 process config
â”œâ”€â”€ requirements.txt           # Production dependencies
â”œâ”€â”€ requirements-dev.txt       # Test dependencies
â”œâ”€â”€ .env.example               # Environment variable template
â”œâ”€â”€ .gitignore
â”œâ”€â”€ FINAL_TEST_REPORT.md       # E2E test results (46/47, 97.9%)
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py                # FastAPI factory + lifespan
â”‚   â”œâ”€â”€ config/
â”‚   â”‚   â”œâ”€â”€ settings.py        # Pydantic Settings (all from .env)
â”‚   â”‚   â””â”€â”€ database.py        # MongoDB + Beanie init
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ diet_session.py    # Session document
â”‚   â”‚   â””â”€â”€ diet_message.py    # Message document
â”‚   â”œâ”€â”€ schemas/
â”‚   â”‚   â”œâ”€â”€ diet.py            # Request / Response schemas
â”‚   â”‚   â””â”€â”€ patient.py         # PatientContext schemas
â”‚   â”œâ”€â”€ services/
â”‚   â”‚   â”œâ”€â”€ llm/
â”‚   â”‚   â”‚   â”œâ”€â”€ base.py        # Abstract LLM + exceptions
â”‚   â”‚   â”‚   â”œâ”€â”€ groq_adapter.py
â”‚   â”‚   â”‚   â”œâ”€â”€ gemini_adapter.py
â”‚   â”‚   â”‚   â”œâ”€â”€ claude_adapter.py
â”‚   â”‚   â”‚   â”œâ”€â”€ openai_adapter.py
â”‚   â”‚   â”‚   â””â”€â”€ router.py      # Multi-provider fallback chain
â”‚   â”‚   â”œâ”€â”€ patient_context.py # Fetch patient data (API + MongoDB fallback)
â”‚   â”‚   â”œâ”€â”€ diet_session.py    # Session CRUD
â”‚   â”‚   â”œâ”€â”€ diet_chat.py       # Orchestration + cuisine switching
â”‚   â”‚   â””â”€â”€ vector_context.py  # Atlas Vector Search (RAG)
â”‚   â”œâ”€â”€ routers/
â”‚   â”‚   â”œâ”€â”€ diet.py            # Diet endpoints
â”‚   â”‚   â””â”€â”€ health.py          # Health + provider status
â”‚   â”œâ”€â”€ middleware/
â”‚   â”‚   â”œâ”€â”€ auth.py            # JWT validation
â”‚   â”‚   â”œâ”€â”€ request_id.py      # X-Request-ID injection
â”‚   â”‚   â”œâ”€â”€ security_headers.py # OWASP security headers
â”‚   â”‚   â””â”€â”€ error_handler.py   # Global exception handlers
â”‚   â””â”€â”€ utils/
â”‚       â”œâ”€â”€ logger.py          # structlog setup
â”‚       â”œâ”€â”€ prompt_builder.py  # Indian cuisine system prompt builder
â”‚       â”œâ”€â”€ provider_state.py  # Quota exhaustion cooldown tracker
â”‚       â””â”€â”€ response_parser.py # Diet plan Markdown parser
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ gen_jwt.py             # Generate a single patient JWT
â”‚   â”œâ”€â”€ _gen_jwts.py           # Generate test JWTs from .env
â”‚   â”œâ”€â”€ _start.py              # Subprocess launcher for server
â”‚   â”œâ”€â”€ _e2e_runner.py         # Full E2E test runner (47 tests)
â”‚   â””â”€â”€ ingest_patient_to_vectors.py  # Atlas Vector ingest
â””â”€â”€ tests/
    â”œâ”€â”€ conftest.py            # Shared fixtures + TEST_JWT_SECRET
    â”œâ”€â”€ test_llm_router.py     # Router + fallback tests
    â”œâ”€â”€ test_diet_session.py   # Session CRUD tests
    â”œâ”€â”€ test_diet_chat.py      # Chat + prompt tests
    â””â”€â”€ test_e2e.py            # Full integration test suite
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

**Best recorded result: 46/47 PASS (97.9%)** â€” see `FINAL_TEST_REPORT.md`.

---

## Security

- JWT validated on every protected endpoint (HS256)
- Patients access only their own sessions (patient-ID-scoped queries)
- Input sanitised against XSS on session creation
- Rate limiting per IP and per patient
- Security response headers on all endpoints
- No hardcoded secrets â€” all config via `.env`
- Swagger/ReDoc UI disabled in production

---

## License

Part of the MediVault Hospital Management System.
