<div align="center">

<img src="https://img.shields.io/badge/AMD_Instinct-MI300X-ED1C24?style=for-the-badge&logo=amd&logoColor=white" />
<img src="https://img.shields.io/badge/ROCm-Stack-ED1C24?style=for-the-badge&logo=amd&logoColor=white" />
<img src="https://img.shields.io/badge/vLLM-Inference-6D28D9?style=for-the-badge" />
<img src="https://img.shields.io/badge/Qwen-Multimodal-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />

<br /><br />

# 🏥 MediAgent

### Autonomous Multi-Agent Medical Imaging Analysis System

**Five specialized AI agents. One radiological verdict. Running entirely on AMD.**

*AMD Developer Hackathon 2026 · Track: Vision & Multimodal AI*

<br />

> Built by **Ramyar** — Security researcher & full-stack developer, Sulaymaniyah, Iraq

</div>

---

## What Is MediAgent?

MediAgent is a production-grade autonomous AI system that analyzes medical images — X-rays, MRI scans, CT scans — through a five-agent pipeline and generates structured, peer-reviewed clinical radiology reports in real time.

Upload an image. Watch five AI agents execute live. Get a formal radiology report with differential diagnoses, ICD-10 codes, a quality score, and a FHIR R4 export ready for any EMR system.

**No cloud APIs. No OpenAI. No Nvidia.**
Pure AMD MI300X inference. Local. Private. Fast.

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMAGE UPLOAD                                 │
│              PNG / JPG / DICOM (.dcm) — up to 20 MB                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │         PARALLEL STAGE          │
          ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│  INTAKE AGENT   │               │  VISION AGENT   │
│                 │               │                 │
│ • Validates     │               │ • Multimodal    │
│   image payload │               │   Qwen analysis │
│ • Normalizes    │               │ • Anatomical    │
│   clinical text │               │   findings      │
│ • Extracts      │               │ • Severity per  │
│   demographics  │               │   region        │
│ • Safety triage │               │ • Confidence    │
│   (16 keywords) │               │   scoring       │
│ • Modality hint │               │ • Anomaly flags │
└────────┬────────┘               └────────┬────────┘
         └──────────────┬──────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │    RESEARCH AGENT     │
            │                       │
            │ • KB cross-reference  │
            │   (15 conditions)     │
            │ • Demographic weight  │
            │ • Ranked differentials│
            │ • ICD-10 codes        │
            │ • Match probabilities │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │     REPORT AGENT      │
            │                       │
            │ • ACR/NICE format     │
            │ • Clinical history    │
            │ • Technique section   │
            │ • Findings narrative  │
            │ • Impression + top Dx │
            │ • Recommendations     │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │     CRITIC AGENT      │
            │                       │
            │ • Cross-validates     │
            │   report vs findings  │
            │ • Quality score 0-100 │
            │ • Uncertainty flags   │
            │ • Disclaimer enforce  │
            └───────────┬───────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FINAL REPORT                                   │
│         Structured JSON · PDF Export · FHIR R4 DiagnosticReport    │
└─────────────────────────────────────────────────────────────────────┘
```

INTAKE and VISION execute **concurrently** — cutting wall-clock latency by running the two most expensive operations in parallel. Everything downstream sequences after both complete.

---

## AMD Hardware Stack

| Component | Technology |
|---|---|
| **GPU** | AMD Instinct MI300X |
| **GPU Software** | ROCm — AMD's open-source GPU compute platform |
| **Inference Server** | vLLM (ROCm build) at `localhost:8000/v1` |
| **Model** | Qwen multimodal — native vision + text |
| **Backend** | FastAPI 0.115 + Uvicorn |
| **Frontend** | Vanilla JS + Tailwind CSS + SSE streaming |

This project is a direct proof of concept that AMD's ROCm stack is **production-viable for real-world medical AI**. Every inference call — vision analysis, clinical normalization, report synthesis, peer review, post-report chat — runs on AMD MI300X. Zero CUDA dependency. Zero cloud API calls.

---

## Key Features

### 🔴 Real-Time SSE Streaming
Watch the pipeline execute live, agent by agent. Every status transition — WAITING → RUNNING → DONE — streams to the dashboard as it happens via Server-Sent Events. Per-agent runtime counters track exactly how long each step takes.

### 👁️ Multimodal Vision Analysis
Qwen processes the raw medical image natively. It returns structured JSON: detected modality, technical quality assessment, per-region findings with anatomical names, radiological descriptions, severity levels (NORMAL / INCIDENTAL / SIGNIFICANT / CRITICAL), confidence scores (0–100), and anomaly flags.

### 🔬 Medical Knowledge Base + ICD-10 Mapping
The Research Agent cross-references vision findings against 15 curated clinical conditions spanning pulmonary, neurological, abdominal, musculoskeletal, and vascular pathology. Every differential diagnosis comes with an ICD-10 code, match probability, and a sentence explaining exactly why the condition matches the findings.

### 🛡️ Critic Agent QA
Every report goes through a peer-review pass before delivery. The Critic checks that all anomalies from the Vision Agent appear in the report, flags low-confidence findings, assigns a quality score (completeness 30% + accuracy 40% + safety 20% + compliance 10%), and hard-caps the score at 40/100 if a core agent failed.

### 🏥 DICOM Support
Upload real `.dcm` files. MediAgent extracts 20+ metadata fields — patient name, study date, institution, modality, body part, KVP, slice thickness, pixel spacing, image dimensions — and pre-populates the intake form automatically. MONOCHROME1 inversion and multi-frame handling included.

### 📋 FHIR R4 Export
Every report can be exported as a fully conformant HL7 FHIR R4 DiagnosticReport resource. Includes an inline Patient resource, Observation resources, LOINC and SNOMED CT codes, severity mapping, full report text in `presentedForm`, and custom extensions for AI quality score and pipeline status. Ready to import into Epic, Cerner, or any FHIR-capable EMR.

### 💬 Post-Report Clinical Chat
After the report is delivered, a ClinicalAdvisorAgent is available for follow-up questions. It answers in 2–4 sentences with direct reference to the report findings. Qwen's thinking/reasoning mode is explicitly disabled — answers are fast, direct, and clinical.

### 🔒 Hard Safety Enforcement
- **16 deterministic safety keywords** — chest pain, stroke symptoms, acute trauma, hemoptysis, sepsis, spinal trauma, and more — trigger urgent flags regardless of LLM output.
- **Age-based alerts** — pediatric (<18) and geriatric (>75) cases are automatically flagged for expert review.
- **Mandatory AI disclaimer** — enforced at two independent layers (Report Agent + Critic Agent) and cannot be bypassed or modified by the LLM.
- **Graceful degradation** — the pipeline produces a report even if individual agents fail, always marking what succeeded and what didn't.

### 📄 Client-Side PDF Export
Full radiology report exported as a formatted PDF directly in the browser using jsPDF — severity color banner, all six report sections, DICOM metadata, QA score. No server round-trip needed.

---

## Agent Architecture

### IntakeAgent
Validates the image payload (minimum size, valid base64), applies deterministic safety triage, and normalizes clinical language. For simple inputs under 120 characters it skips the LLM entirely and uses a built-in layman-to-medical term map (22 entries: "can't breathe" → "dyspnea", "lump" → "mass/nodule", "dizzy" → "dizziness/vertigo", etc.). Only calls the LLM for complex clinical narratives with comorbidities or medical history. Falls back cleanly to raw input preservation if the LLM is unavailable.

### VisionAgent
Sends the base64 image and clinical context to Qwen at temperature 0.0 with a strict JSON schema enforced via system prompt. Handles malformed enum values from the LLM with safe conversion fallbacks — a single bad field never drops a finding. Tracks token usage and anomaly counts in the output metadata.

### ResearchAgent
Pre-filters the knowledge base to only conditions compatible with the detected modality before sending to the LLM — reducing prompt size and improving accuracy. Enforces strict output rules: only conditions from the KB, 2–4 differentials maximum, 5% minimum probability, exact ICD-10 codes, and evidence sentences that actually explain the match.

### ReportAgent
Builds a structured prompt with clearly labeled sections — clinical history, imaging technique, findings block, differentials block — and asks the LLM to synthesize them into a formal ACR/NICE radiology report. The disclaimer is overwritten to the exact regulatory string after LLM generation, unconditionally.

### CriticAgent
Operates at temperature 0.0 for fully deterministic QA. Receives the draft report and the full pipeline state including raw vision findings. Checks every anomaly is accounted for, flags low-confidence observations, and appends a `[QUALITY ASSESSMENT]` block to the recommendations section with score, issues, and uncertainty warnings.

### ClinicalAdvisorAgent
Activated only after report delivery, scoped to the specific report's findings. Strips all Qwen thinking output via multi-layer regex before returning the answer — handles `<think>` XML blocks, markdown think fences, and plain-text reasoning preambles.

---

## LLM Client

The `LLMClient` wraps the OpenAI Python SDK pointed at the local vLLM endpoint. It handles:

- Text completions with optional JSON mode enforcement
- Multimodal completions with base64 image injection
- Token-level streaming with an `on_token` callback
- 3-attempt retry loop with 1-second flat backoff
- 90-second timeout per call
- Dual-strategy JSON extraction: direct parse first, then character-by-character brace-matching fallback for responses where the LLM adds conversational padding

---

## Medical Knowledge Base

15 conditions covering the most common radiological findings across all supported modalities:

| Condition | ICD-10 | Modalities | Severity |
|---|---|---|---|
| Community-Acquired Pneumonia | J18.9 | X-RAY, CT | SIGNIFICANT |
| Cardiogenic Pulmonary Edema | J81.0 | X-RAY, CT | CRITICAL |
| Pleural Effusion | J90 | X-RAY, CT, MRI | SIGNIFICANT |
| Spontaneous Pneumothorax | J93.9 | X-RAY, CT | CRITICAL |
| Intracerebral Hemorrhage | I61.9 | CT, MRI | CRITICAL |
| Ischemic Stroke | I63.9 | CT, MRI | CRITICAL |
| Intracranial Neoplasm | C71.9 | MRI, CT | SIGNIFICANT |
| Abdominal Aortic Aneurysm | I71.4 | CT, MRI | CRITICAL |
| Nephrolithiasis | N20.0 | CT, X-RAY | SIGNIFICANT |
| Small Bowel Obstruction | K56.6 | X-RAY, CT | SIGNIFICANT |
| Long Bone Fracture | S82.902 | X-RAY, CT | SIGNIFICANT |
| Degenerative Joint Disease | M19.90 | X-RAY, MRI | INCIDENTAL |
| Hepatic Steatosis | K76.0 | CT, MRI | INCIDENTAL |
| Herniated Disc | M51.16 | MRI, CT | SIGNIFICANT |
| Pulmonary Nodule | R91.1 | X-RAY, CT | SIGNIFICANT |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Clinical dashboard UI |
| `GET` | `/health` | System health, version, active sessions |
| `GET` | `/metrics/gpu` | Live AMD GPU metrics (util, VRAM, temp, power) |
| `POST` | `/analyze` | Synchronous pipeline → full JSON report |
| `POST` | `/analyze/stream` | Real-time SSE streaming pipeline |
| `GET` | `/status/{report_id}` | Poll live pipeline state |
| `POST` | `/chat/{report_id}` | Post-report clinical Q&A |
| `GET` | `/api/docs` | Swagger UI |
| `GET` | `/api/redoc` | ReDoc UI |

### `/analyze/stream` — SSE Event Types

```json
// Agent status update (emitted on every state transition)
{"agent": "VISION", "status": "RUNNING"}
{"agent": "VISION", "status": "DONE"}

// Final report (emitted when pipeline completes)
{"type": "report", "data": {...}, "report_id": "REP-A3F9C2D1B4E7"}

// Error
{"type": "error", "message": "Pipeline produced no report"}
```

### Form Fields (`/analyze`, `/analyze/stream`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | File | ✅ | PNG, JPG, or DICOM (.dcm), max 20 MB |
| `symptoms` | string | — | Free-text chief complaint |
| `age` | integer | — | 0–120 |
| `sex` | string | — | `M`, `F`, or `O` |
| `clinical_context` | string | — | Medical history, referral details |

---

## Data Models

```
PatientInput
    └── image_base64, symptoms, age, sex, clinical_context

PipelineState
    ├── agent_statuses: {INTAKE, VISION, RESEARCH, REPORT, CRITIC}
    ├── intake_output: IntakeOutput
    ├── vision_output: VisionOutput
    │       └── findings: [VisionFinding, ...]
    │               └── anatomical_region, description, severity,
    │                   confidence, confidence_score, is_anomaly
    ├── research_output: ResearchOutput
    │       └── differential_diagnoses: [KnowledgeMatch, ...]
    │               └── condition_name, match_probability,
    │                   supporting_evidence, differential_rank, icd10_code
    ├── report_draft: ReportSection
    │       └── clinical_history, technique, findings, impression,
    │           recommendations, disclaimer
    └── final_report: FinalReport
            └── report_id, patient_metadata, sections, vision_summary,
                research_summary, overall_severity, agent_pipeline_status,
                generation_timestamp
```

---

## Project Structure

```
mediagent/
├── main.py                  ← FastAPI server, all routes, SSE orchestration
├── core/
│   ├── llm.py               ← LLM client (retry, vision, streaming, JSON extraction)
│   ├── models.py            ← All Pydantic v2 data models
│   ├── pipeline.py          ← Parallel pipeline orchestrator
│   ├── dicom.py             ← DICOM parser (pydicom + numpy + Pillow)
│   └── fhir.py              ← FHIR R4 DiagnosticReport builder
├── agents/
│   ├── intake.py            ← Input validation, normalization, safety triage
│   ├── vision.py            ← Multimodal image analysis
│   ├── research.py          ← KB matching, ICD-10, differential diagnosis
│   ├── report.py            ← ACR/NICE radiology report synthesis
│   ├── critic.py            ← QA validation, quality scoring
│   └── advisor.py           ← Post-report clinical Q&A
├── static/
│   └── index.html           ← Full dashboard (Tailwind + Chart.js + SSE)
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- vLLM running a Qwen multimodal model on ROCm, accessible at `http://localhost:8000/v1`
- ROCm-compatible AMD GPU (MI300X recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ramyar2007/mediagent
cd mediagent

# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set LLM_BASE_URL to your vLLM endpoint
```

### Environment Variables

```env
LLM_BASE_URL=http://localhost:8000/v1   # vLLM OpenAI-compatible endpoint
LLM_MODEL=/model                         # Model path served by vLLM
APP_PORT=8090                            # Server port
```

### Run

```bash
python main.py
```

Dashboard available at **http://localhost:8090**

Swagger docs at **http://localhost:8090/api/docs**

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.6 | Web framework |
| `uvicorn[standard]` | 0.34.0 | ASGI server |
| `openai` | 1.58.1 | SDK for vLLM OpenAI-compatible API |
| `python-multipart` | 0.0.20 | Multipart form / file upload |
| `pydantic` | 2.10.5 | Data validation and serialization |
| `Pillow` | 11.1.0 | Image processing for DICOM conversion |
| `pydicom` | 2.4.4 | DICOM file parsing and metadata extraction |
| `numpy` | 1.26.4 | Pixel array normalization for DICOM |

Optional: `amdsmi` Python library — used automatically when available for more accurate GPU metrics than the `rocm-smi` CLI fallback.

---

## Clinical Safety

MediAgent is built with clinical safety as a first-class concern, not an afterthought.

**Mandatory disclaimer** — enforced at two independent code layers and cannot be overridden by any LLM output:

> *"This analysis is AI-generated and must be reviewed by a licensed radiologist before any clinical decisions are made."*

**Hard safety rules that run deterministically, without LLM involvement:**
- 16 urgent clinical keywords trigger immediate flags before any AI processing
- Pediatric and geriatric age thresholds auto-flag for specialist review
- Quality score is hard-capped at 40/100 if core agents (Vision, Report) fail
- Low-confidence findings are always flagged with confirmatory imaging recommendations
- The disclaimer is re-enforced after every LLM call, unconditionally

**This system is a decision support tool, not a clinical decision maker.** Every output is intended to assist, not replace, a licensed radiologist.

---

## Dashboard Preview

The single-page clinical dashboard provides:

- **Live pipeline panel** — real-time agent status cards with per-step runtime counters
- **Analytics tab** — severity distribution donut chart, differential diagnosis confidence bar chart, agent timing bar chart — all populated from structured model output
- **Report panel** — severity banner, safety flags, all six report sections, finding cards color-coded by severity
- **DICOM metadata card** — study date, institution, modality, body part, technical parameters
- **PDF export** — full formatted report generated client-side
- **Clinical chat** — slide-up Q&A panel backed by the ClinicalAdvisorAgent
- **AMD GPU panel** — live util %, VRAM used/total, temperature, power draw — polling every 3 seconds

---

## Built For

**AMD Developer Hackathon 2026**
Track: Vision & Multimodal AI

This project demonstrates that AMD's ROCm ecosystem is a complete, production-viable alternative for serious AI workloads. Medical imaging analysis — with real multimodal vision, structured clinical reasoning, and standards-compliant output — running fully on AMD MI300X without a single NVIDIA or cloud dependency.

---

<div align="center">

**Built by Ramyar · Sulaymaniyah, Iraq**

*#AMDDevChallenge · AMD Instinct MI300X · ROCm · vLLM · Qwen*

</div>
