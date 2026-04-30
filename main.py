# mediagent/main.py
"""
MediAgent - Autonomous Multi-Agent Medical Imaging Analysis System
Production FastAPI Server & Orchestrator Entry Point
Runs on AMD MI300X infrastructure with local Qwen inference.
Includes real-time SSE streaming for live pipeline tracking.
"""

import logging
import base64
import uuid
import json
import asyncio
import uvicorn
from datetime import datetime
from typing import Dict, Optional, Any, AsyncGenerator, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.llm import LLMClient
from core.models import PatientInput, PipelineState
from core.pipeline import PipelineOrchestrator
from agents.intake import IntakeAgent
from agents.vision import VisionAgent
from agents.research import ResearchAgent
from agents.report import ReportAgent
from agents.critic import CriticAgent

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("mediagent.server")

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediAgent API",
    version="1.0.0",
    description="Autonomous Multi-Agent Medical Imaging Analysis System",
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

# In-memory registry for pipeline state tracking & status polling
pipeline_registry: Dict[str, PipelineState] = {}


# ─────────────────────────────────────────────────────────────────────────────
# LIFECYCLE & DEPENDENCY INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize LLM client, specialized agents, and orchestration pipeline."""
    logger.info("🚀 MediAgent System Startup Sequence Initiated")
    
    try:
        llm_client = LLMClient()
        logger.info("✅ LLM Client connected to local endpoint")
        
        # Store components in app.state for reuse across endpoints
        app.state.llm_client = llm_client
        app.state.intake_agent = IntakeAgent(llm_client=llm_client)
        app.state.vision_agent = VisionAgent(llm_client=llm_client)
        app.state.research_agent = ResearchAgent(llm_client=llm_client)
        app.state.report_agent = ReportAgent(llm_client=llm_client)
        app.state.critic_agent = CriticAgent(llm_client=llm_client)
        
        def default_status_callback(state: PipelineState):
            report_id = [k for k, v in pipeline_registry.items() if v is state]
            if report_id:
                logger.info(f"🔄 [{report_id[0]}] Pipeline Step: {state.current_step}")
        
        app.state.orchestrator = PipelineOrchestrator(
            intake_agent=app.state.intake_agent,
            vision_agent=app.state.vision_agent,
            research_agent=app.state.research_agent,
            report_agent=app.state.report_agent,
            critic_agent=app.state.critic_agent,
            on_status_update=default_status_callback
        )
        
        logger.info("✅ Pipeline Orchestrator Ready. System Online.")
        
    except Exception as e:
        logger.critical(f"💥 Startup failure: {e}")
        raise SystemExit(f"Critical initialization error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main clinical dashboard interface."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """System health & infrastructure verification endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": "MediAgent v1.0.0",
        "infrastructure": "AMD MI300X Local Inference",
        "model_endpoint": "http://localhost:8000/v1",
        "agents_loaded": app.state.orchestrator is not None,
        "active_sessions": len(pipeline_registry)
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(..., description="Medical image file (PNG/JPG)"),
    symptoms: str = Form(default="", description="Patient chief complaint or symptoms"),
    age: Optional[int] = Form(default=None, ge=0, le=120, description="Patient age"),
    sex: Optional[str] = Form(default=None, description="Patient biological sex (M/F/O)"),
    clinical_context: str = Form(default="", description="Additional medical history")
):
    """
    Primary diagnostic endpoint (Synchronous).
    Accepts multipart form data, converts image to base64, executes the full 
    5-agent pipeline, and returns a structured FinalReport as JSON.
    """
    logger.info(f"📥 [SYNC] New analysis request received | Content-Type: {image.content_type}")
    report_id = f"REP-{uuid.uuid4().hex[:12].upper()}"
    
    try:
        image_bytes = await image.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds 20MB size limit.")
            
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        base64_image = f"data:{content_type};base64,{base64_data}"
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    patient_input = PatientInput(
        image_base64=base64_image,
        symptoms=symptoms,
        age=age,
        sex=sex,
        clinical_context=clinical_context
    )
    
    pipeline_registry[report_id] = PipelineState()
    
    try:
        orchestrator = app.state.orchestrator
        state = orchestrator.run(patient_input)
        pipeline_registry[report_id] = state
        
        if not state.final_report:
            logger.error("⚠️ Pipeline completed but failed to generate final report.")
            raise HTTPException(status_code=500, detail="Report generation failed. Check logs.")
            
        logger.info(f"✅ [SYNC] Analysis complete | Report ID: {report_id}")
        return JSONResponse(content=state.final_report.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"💥 Pipeline execution failed: {e}")
        pipeline_registry[report_id].error_log.append(f"SYSTEM_CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {str(e)}")


@app.post("/analyze/stream")
async def analyze_stream(
    image: UploadFile = File(..., description="Medical image file (PNG/JPG)"),
    symptoms: str = Form(default="", description="Patient chief complaint or symptoms"),
    age: Optional[int] = Form(default=None, ge=0, le=120, description="Patient age"),
    sex: Optional[str] = Form(default=None, description="Patient biological sex (M/F/O)"),
    clinical_context: str = Form(default="", description="Additional medical history")
):
    """
    Real-time SSE streaming endpoint.
    Accepts same multipart form data as /analyze but streams JSON events via
    text/event-stream as each agent completes. Final event contains the full report.
    """
    logger.info(f"📥 [STREAM] New streaming analysis request received")
    
    # 1. Parse form data & encode image
    try:
        image_bytes = await image.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds 20MB size limit.")
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        base64_image = f"data:{content_type};base64,{base64_data}"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    patient_input = PatientInput(
        image_base64=base64_image,
        symptoms=symptoms,
        age=age,
        sex=sex,
        clinical_context=clinical_context
    )

    # 2. Async generator for SSE events
    async def event_generator() -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        executor = ThreadPoolExecutor(max_workers=1)

        # Callback pushes status updates into the async queue
        def on_stream_update(state: PipelineState) -> None:
            for agent_name, status in state.agent_statuses.items():
                payload = {"agent": agent_name, "status": status.value}
                queue.put_nowait(f"data: {json.dumps(payload, default=str)}\n\n")

        # Run synchronous pipeline in background thread
        def run_pipeline_threaded() -> None:
            try:
                # Fresh orchestrator instance for isolated stream context
                orchestrator = PipelineOrchestrator(
                    intake_agent=app.state.intake_agent,
                    vision_agent=app.state.vision_agent,
                    research_agent=app.state.research_agent,
                    report_agent=app.state.report_agent,
                    critic_agent=app.state.critic_agent,
                    on_status_update=on_stream_update
                )
                state = orchestrator.run(patient_input)
                
                if state.final_report:
                    payload = {"type": "report", "data": state.final_report.model_dump()}
                    queue.put_nowait(f"data: {json.dumps(payload, default=str)}\n\n")
                else:
                    payload = {"type": "error", "message": "Pipeline completed without final report"}
                    queue.put_nowait(f"data: {json.dumps(payload, default=str)}\n\n")
            except Exception as e:
                logger.exception(f"💥 Stream pipeline crashed: {e}")
                payload = {"type": "error", "message": str(e)}
                queue.put_nowait(f"data: {json.dumps(payload, default=str)}\n\n")
            finally:
                queue.put_nowait(None)  # Sentinel to terminate generator

        # Launch pipeline in thread pool
        asyncio.get_running_loop().run_in_executor(executor, run_pipeline_threaded)

        # Yield events until sentinel received
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/status/{report_id}")
async def get_pipeline_status(report_id: str):
    """Real-time status polling endpoint for frontend pipeline visualization."""
    if report_id not in pipeline_registry:
        raise HTTPException(status_code=404, detail="Report ID not found or expired.")
        
    state = pipeline_registry[report_id]
    return {
        "report_id": report_id,
        "current_step": state.current_step,
        "agent_statuses": {k: v.value for k, v in state.agent_statuses.items()},
        "error_log": state.error_log,
        "completed": state.current_step == "COMPLETE"
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILE MOUNTING
# ─────────────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🏥 Starting MediAgent Clinical Dashboard & API Server on port 8080")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8090,
        log_level="info",
        reload=False
    )
