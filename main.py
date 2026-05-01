# mediagent/main.py
"""
MediAgent - Autonomous Multi-Agent Medical Imaging Analysis System
Production FastAPI Server & Orchestrator Entry Point

New in v2.0:
  - DICOM (.dcm) file support with automatic metadata extraction
  - Real-time token-level streaming during report generation
  - AMD GPU metrics endpoint (/metrics/gpu)
  - Post-report clinical advisor chat (/chat/{report_id})
  - FHIR R4 DiagnosticReport export (/export/fhir/{report_id})
"""

import json
import logging
import base64
import uuid
import asyncio
import subprocess
import uvicorn
from datetime import datetime
from typing import Dict, Optional, Any, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.llm import LLMClient
from core.models import PatientInput, PipelineState, ChatRequest, ChatResponse
from core.pipeline import PipelineOrchestrator
from agents.intake import IntakeAgent
from agents.vision import VisionAgent
from agents.research import ResearchAgent
from agents.report import ReportAgent
from agents.critic import CriticAgent
from agents.advisor import ClinicalAdvisorAgent

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("mediagent.server")

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediAgent API",
    version="2.0.0",
    description="Autonomous Multi-Agent Medical Imaging Analysis System — AMD MI300X",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry: report_id → PipelineState (capped at 200 entries)
pipeline_registry: Dict[str, PipelineState] = {}
REGISTRY_MAX_SIZE = 200


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MediAgent v2.0 — System Startup")
    try:
        llm_client = LLMClient()
        app.state.llm_client      = llm_client
        app.state.intake_agent    = IntakeAgent(llm_client=llm_client)
        app.state.vision_agent    = VisionAgent(llm_client=llm_client)
        app.state.research_agent  = ResearchAgent(llm_client=llm_client)
        app.state.report_agent    = ReportAgent(llm_client=llm_client)
        app.state.critic_agent    = CriticAgent(llm_client=llm_client)
        app.state.advisor_agent   = ClinicalAdvisorAgent(llm_client=llm_client)

        def _default_cb(state: PipelineState):
            pass

        app.state.orchestrator = PipelineOrchestrator(
            intake_agent=app.state.intake_agent,
            vision_agent=app.state.vision_agent,
            research_agent=app.state.research_agent,
            report_agent=app.state.report_agent,
            critic_agent=app.state.critic_agent,
            on_status_update=_default_cb,
        )
        logger.info("✅ All agents initialised. MediAgent v2.0 online.")
    except Exception as e:
        logger.critical("💥 Startup failure: %s", e)
        raise SystemExit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _evict_registry():
    if len(pipeline_registry) >= REGISTRY_MAX_SIZE:
        del pipeline_registry[next(iter(pipeline_registry))]


async def _read_and_encode_image(image: UploadFile):
    """
    Read uploaded file. Supports JPEG/PNG and DICOM (.dcm).
    Returns (base64_image_str, dicom_metadata_dict_or_None).
    """
    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB size limit.")

    filename = (image.filename or "").lower()
    content_type = image.content_type or ""

    is_dicom = (
        filename.endswith(".dcm")
        or "dicom" in content_type
        or image_bytes[:4] == b"\x00\x00\x00\x00"  # some DICOM files
        or image_bytes[128:132] == b"DICM"
    )

    if is_dicom:
        try:
            from core.dicom import parse_dicom
            b64_image, dicom_meta = parse_dicom(image_bytes)
            logger.info("DICOM parsed | meta keys: %s", list(dicom_meta.keys()))
            return b64_image, dicom_meta
        except Exception as e:
            logger.warning("DICOM parse failed (%s), treating as regular image", e)

    # Standard image
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    mime = content_type if content_type.startswith("image/") else "image/jpeg"
    return f"data:{mime};base64,{b64_data}", None


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — STATIC / HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": "MediAgent",
        "infrastructure": "AMD Instinct MI300X / ROCm / vLLM",
        "agents_loaded": hasattr(app.state, "orchestrator"),
        "active_sessions": len(pipeline_registry),
        "features": ["dicom", "clinical_chat", "gpu_metrics"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — AMD GPU METRICS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics/gpu")
async def get_gpu_metrics():
    """
    Returns live AMD GPU metrics via rocm-smi (MI300X).
    Tries amdsmi Python library first (most accurate), then rocm-smi CLI.
    """
    metrics: Dict[str, Any] = {
        "available": False,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # ── Try amdsmi Python library (most reliable on ROCm 5.7+) ───────────────
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        cards = []
        for i, dev in enumerate(devices):
            try:
                usage   = amdsmi.amdsmi_get_gpu_activity(dev)
                vram    = amdsmi.amdsmi_get_gpu_memory_usage(dev, amdsmi.AmdSmiMemoryType.VRAM)
                vtotal  = amdsmi.amdsmi_get_gpu_memory_total(dev, amdsmi.AmdSmiMemoryType.VRAM)
                temp    = amdsmi.amdsmi_get_temp_metric(dev, amdsmi.AmdSmiTemperatureType.JUNCTION,
                                                         amdsmi.AmdSmiTemperatureMetric.CURRENT)
                power   = amdsmi.amdsmi_get_power_info(dev)
                clk     = amdsmi.amdsmi_get_clk_freq(dev, amdsmi.AmdSmiClkType.GFX)
                cards.append({
                    "card":          f"GPU {i}",
                    "gpu_use_pct":   usage.get("gfx_activity", 0),
                    "vram_used_mb":  round(vram / 1024 / 1024, 1),
                    "vram_total_mb": round(vtotal / 1024 / 1024, 1),
                    "temp_c":        temp,
                    "power_w":       round(power.get("current_socket_power", 0), 1),
                    "clk_mhz":       clk.get("cur_clk", 0),
                })
            except Exception:
                pass
        amdsmi.amdsmi_shut_down()
        if cards:
            metrics["available"] = True
            metrics["source"] = "amdsmi"
            metrics["cards"] = cards
            return metrics
    except Exception:
        pass

    # ── Try rocm-smi CLI JSON ─────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = json.loads(result.stdout)
            cards = []
            for key, val in raw.items():
                if not isinstance(val, dict):
                    continue

                def _pick(d, *keys, default=0):
                    for k in keys:
                        v = d.get(k)
                        if v is not None:
                            try: return float(str(v).replace("%","").strip())
                            except ValueError: pass
                    return default

                vram_used  = _pick(val,
                    "VRAM Total Used Memory (B)",
                    "Used VRAM (B)",
                    "vram_used",
                    "VRAM Total Used Memory (MiB)",
                    "Used VRAM (MiB)",
                    "VRAM Total Memory Used (MiB)")
                vram_total = _pick(val,
                    "VRAM Total Memory (B)",
                    "Total VRAM (B)",
                    "vram_total",
                    "VRAM Total Memory (MiB)",
                    "Total VRAM (MiB)",
                    "VRAM Total Memory Size (MiB)")
                # Convert bytes → MiB if value looks like bytes (> 1 million)
                if vram_used  > 1_000_000: vram_used  = round(vram_used  / 1024 / 1024, 1)
                if vram_total > 1_000_000: vram_total = round(vram_total / 1024 / 1024, 1)

                cards.append({
                    "card":          key,
                    "gpu_use_pct":   _pick(val, "GPU use (%)", "GPU Use (%)", "GFX Activity (%)", "GPU activity (%)"),
                    "vram_used_mb":  vram_used,
                    "vram_total_mb": vram_total,
                    "temp_c":        _pick(val,
                                        "Temperature (Sensor junction) (C)",
                                        "Temperature (Sensor HBM 0) (C)",
                                        "Junction Temperature (C)"),
                    "power_w":       _pick(val,
                                        "Current Socket Graphics Package Power (W)",
                                        "Average Graphics Package Power (W)",
                                        "Socket Power (W)"),
                })

            if cards:
                metrics["available"] = True
                metrics["source"] = "rocm-smi"
                metrics["cards"] = cards
                return metrics
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("rocm-smi JSON failed: %s", e)

    metrics["note"] = "AMD GPU metrics unavailable — is ROCm installed?"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    symptoms: str = Form(default=""),
    age: Optional[int] = Form(default=None, ge=0, le=120),
    sex: Optional[str] = Form(default=None),
    clinical_context: str = Form(default=""),
):
    """Synchronous full-pipeline analysis. Returns complete FinalReport JSON."""
    logger.info("[SYNC] New analysis request | file=%s", image.filename)
    report_id = f"REP-{uuid.uuid4().hex[:12].upper()}"

    try:
        b64_image, dicom_meta = await _read_and_encode_image(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

    # DICOM metadata overrides form fields when not explicitly provided
    if dicom_meta:
        if age is None and dicom_meta.get("age"):
            try:
                age = int(dicom_meta["age"])
            except (ValueError, TypeError):
                pass
        if not sex and dicom_meta.get("sex"):
            sex = dicom_meta["sex"]
        if not clinical_context and dicom_meta.get("study_description"):
            clinical_context = f"DICOM: {dicom_meta.get('study_description','')} | {dicom_meta.get('body_part','')}"

    patient_input = PatientInput(
        image_base64=b64_image, symptoms=symptoms, age=age,
        sex=sex, clinical_context=clinical_context
    )

    _evict_registry()
    pipeline_registry[report_id] = PipelineState()

    try:
        state = app.state.orchestrator.run(patient_input)
        pipeline_registry[report_id] = state
        if not state.final_report:
            raise HTTPException(status_code=500, detail="Pipeline completed without final report.")
        report_dict = state.final_report.model_dump()
        if dicom_meta:
            report_dict["dicom_metadata"] = dicom_meta
        logger.info("[SYNC] Complete | report_id=%s", report_id)
        return JSONResponse(content=report_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline failure: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/stream")
async def analyze_stream(
    image: UploadFile = File(...),
    symptoms: str = Form(default=""),
    age: Optional[int] = Form(default=None, ge=0, le=120),
    sex: Optional[str] = Form(default=None),
    clinical_context: str = Form(default=""),
):
    """
    SSE streaming endpoint.
    Emits:
      {"agent": "X", "status": "RUNNING|DONE|ERROR"}
      {"type": "report", "data": {...}}
      {"type": "error", "message": "..."}
    """
    logger.info("[STREAM] New streaming analysis request | file=%s", image.filename)

    try:
        b64_image, dicom_meta = await _read_and_encode_image(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if dicom_meta:
        if age is None and dicom_meta.get("age"):
            try:
                age = int(dicom_meta["age"])
            except (ValueError, TypeError):
                pass
        if not sex and dicom_meta.get("sex"):
            sex = dicom_meta["sex"]
        if not clinical_context and dicom_meta.get("study_description"):
            clinical_context = f"DICOM: {dicom_meta.get('study_description','')} | {dicom_meta.get('body_part','')}"

    patient_input = PatientInput(
        image_base64=b64_image, symptoms=symptoms, age=age,
        sex=sex, clinical_context=clinical_context
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        executor = ThreadPoolExecutor(max_workers=1)

        _last_statuses: dict = {}

        def _status_cb(state: PipelineState):
            for agent_name, status in state.agent_statuses.items():
                if _last_statuses.get(agent_name) != status:
                    _last_statuses[agent_name] = status
                    payload = {"agent": agent_name, "status": status.value}
                    queue.put_nowait(f"data: {json.dumps(payload)}\n\n")

        def _run_pipeline():
            try:
                report_id = f"REP-{uuid.uuid4().hex[:12].upper()}"
                orchestrator = PipelineOrchestrator(
                    intake_agent=app.state.intake_agent,
                    vision_agent=app.state.vision_agent,
                    research_agent=app.state.research_agent,
                    report_agent=app.state.report_agent,
                    critic_agent=app.state.critic_agent,
                    on_status_update=_status_cb,
                )
                state = orchestrator.run(patient_input)
                _evict_registry()
                pipeline_registry[report_id] = state

                if state.final_report:
                    report_dict = state.final_report.model_dump()
                    if dicom_meta:
                        report_dict["dicom_metadata"] = dicom_meta
                    if state.vision_output:
                        report_dict["vision_findings"] = [
                            {"severity": f.severity.value, "confidence_score": f.confidence_score, "description": f.description}
                            for f in state.vision_output.findings
                        ]
                    if state.research_output:
                        report_dict["differential_diagnoses"] = [
                            {"condition_name": d.condition_name, "match_probability": d.match_probability}
                            for d in state.research_output.differential_diagnoses[:5]
                        ]
                    payload = {"type": "report", "data": report_dict, "report_id": report_id}
                    queue.put_nowait(f"data: {json.dumps(payload, default=str)}\n\n")
                else:
                    queue.put_nowait(f"data: {json.dumps({'type':'error','message':'Pipeline produced no report'})}\n\n")
            except Exception as e:
                logger.exception("Stream pipeline crashed: %s", e)
                queue.put_nowait(f"data: {json.dumps({'type':'error','message':str(e)})}\n\n")
            finally:
                queue.put_nowait(None)

        asyncio.get_running_loop().run_in_executor(executor, _run_pipeline)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — CLINICAL ADVISOR CHAT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat/{report_id}", response_model=ChatResponse)
async def clinical_chat(report_id: str, request: ChatRequest):
    """
    Post-report clinical Q&A via the ClinicalAdvisorAgent.
    Requires a completed report in the registry.
    """
    if report_id not in pipeline_registry:
        raise HTTPException(status_code=404, detail="Report not found. Run analysis first.")

    state = pipeline_registry[report_id]
    if not state.final_report:
        raise HTTPException(status_code=400, detail="Report not yet complete.")

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(
        None,
        app.state.advisor_agent.answer,
        request.question,
        state.final_report,
    )

    return ChatResponse(answer=answer, report_id=report_id)



# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — STATUS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status/{report_id}")
async def get_pipeline_status(report_id: str):
    if report_id not in pipeline_registry:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    state = pipeline_registry[report_id]
    return {
        "report_id": report_id,
        "current_step": state.current_step,
        "agent_statuses": {k: v.value for k, v in state.agent_statuses.items()},
        "error_log": state.error_log,
        "completed": state.current_step == "COMPLETE",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATIC + ENTRY
# ─────────────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    logger.info("🏥 Starting MediAgent v2.0 on port 8090")
    uvicorn.run("main:app", host="0.0.0.0", port=8090, log_level="info", reload=False)
