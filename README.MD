# 🏥 MediAgent — Autonomous Multi-Agent Medical Imaging Analysis

> Built for the AMD Developer Hackathon 2026 | Track: Vision & Multimodal AI
> Powered by AMD Instinct MI300X · ROCm · vLLM · Qwen

---

## What Is MediAgent?

MediAgent is a **production-ready autonomous AI system** that analyzes medical images (X-rays, MRI, CT scans) through a 5-agent pipeline and generates structured clinical radiology reports — in real time, running entirely on AMD hardware.

No cloud APIs. No OpenAI. No Nvidia.
**Pure AMD MI300X inference. Local. Private. Fast.**

---

## Live Demo

> Deployed on AMD Developer Cloud — AMD Instinct MI300X GPU
> Running vLLM + Qwen (multimodal) via ROCm stack

---

## The 5-Agent Pipeline

```
📥 Image Upload
      │
      ▼
┌─────────────┐
│ INTAKE Agent│  → Validates image, extracts patient demographics,
│             │    infers modality (X-RAY / MRI / CT), safety triage
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ VISION Agent│  → Multimodal LLM image analysis → anatomical findings,
│             │    severity (NORMAL / INCIDENTAL / SIGNIFICANT / CRITICAL),
│             │    confidence scores, anomaly detection
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ RESEARCH Agent   │  → Cross-references findings against medical knowledge
│                  │    base → ICD-10 codes, differential diagnoses,
│                  │    match probabilities
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ REPORT Agent│  → Synthesizes all upstream data into a formal radiology
│             │    report (ACR/NICE conventions): clinical history,
│             │    technique, findings, impression, recommendations
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ CRITIC Agent│  → QA/QC validation: cross-checks report vs findings,
│             │    flags low-confidence observations, assigns quality
│             │    score (0–100), enforces regulatory disclaimers
└──────┬──────┘
       │
       ▼
📄 Final Structured Report (PDF Export)
```

---

## AMD Hardware & Stack

| Component | Technology |
|-----------|-----------|
| **GPU** | AMD Instinct MI300X |
| **GPU Software** | ROCm (AMD's open-source GPU compute platform) |
| **Inference Server** | vLLM (ROCm build) |
| **Model** | Qwen (multimodal — vision + text) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Tailwind CSS + Vanilla JS + SSE streaming |

The entire inference stack runs on **AMD Developer Cloud** using MI300X GPUs. No CUDA. No Nvidia dependency. This project proves AMD's ROCm stack is production-viable for real-world medical AI workloads.

---

## Key Features

- **Real-time SSE streaming** — watch each agent execute live in the dashboard
- **Multimodal vision analysis** — Qwen processes medical images natively
- **Graceful degradation** — pipeline continues even if individual agents fail
- **Safety triage** — 16 deterministic keywords (chest pain, stroke, trauma, etc.) trigger urgent flags
- **Age-based alerts** — pediatric (<18) and geriatric (>75) cases automatically flagged
- **ICD-10 mapping** — findings mapped to standard diagnostic codes
- **Quality scoring** — Critic agent validates every report before delivery
- **PDF export** — client-side report generation with severity color-coding
- **Hard-enforced disclaimer** — every report includes regulatory AI disclaimer

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Clinical dashboard UI |
| `/analyze` | POST | Synchronous full pipeline → JSON report |
| `/analyze/stream` | POST | Real-time SSE streaming pipeline |
| `/status/{report_id}` | GET | Live pipeline state polling |
| `/health` | GET | System health + infrastructure status |
| `/api/docs` | GET | Swagger UI |

**Form fields:** `image` (PNG/JPG ≤ 20MB), `symptoms`, `age`, `sex`, `clinical_context`

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/Ramyar2007/mediagent
cd mediagent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Set LLM_BASE_URL to your vLLM endpoint (AMD MI300X or local)

# 4. Start the server
python main.py
# Dashboard at http://localhost:8090
```

**Requirements:** Python 3.12+, vLLM running Qwen on ROCm

---

## Environment Variables

```env
LLM_BASE_URL=http://localhost:8000/v1   # vLLM endpoint (AMD MI300X)
LLM_MODEL=/model                         # Model path served by vLLM
APP_PORT=8090                            # FastAPI server port
```

---

## Project Structure

```
mediagent/
├── main.py                  # FastAPI server, routes, SSE streaming
├── core/
│   ├── llm.py               # LLM client (retry logic, vision support)
│   ├── models.py            # Pydantic models for all data types
│   └── pipeline.py          # Pipeline orchestrator
├── agents/
│   ├── intake.py            # Patient intake + safety triage
│   ├── vision.py            # Multimodal image analysis
│   ├── research.py          # Medical KB + ICD-10 matching
│   ├── report.py            # Radiology report generation
│   └── critic.py            # QA validation + quality scoring
├── static/
│   └── index.html           # Clinical dashboard (Tailwind + SSE)
├── requirements.txt
└── .env.example
```

---

## Safety & Clinical Compliance

- Every report includes a mandatory AI disclaimer: *"This analysis is AI-generated and must be reviewed by a licensed radiologist before any clinical decisions are made."*
- Critic agent hard-enforces disclaimer — cannot be bypassed
- Quality score capped at 40/100 if core agents fail
- All findings cross-validated before report delivery

---

## Built By

**Ramyar** — Security researcher & full-stack developer, Slemani, Iraq
AMD Developer Hackathon 2026 · Track: Vision & Multimodal AI

---

*#AMDDevChallenge — Built on AMD MI300X · ROCm · vLLM · Qwen*
