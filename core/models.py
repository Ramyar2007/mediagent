# mediagent/core/models.py
"""
Pydantic data models for MediAgent multi-agent medical imaging pipeline.
Defines input, agent outputs, report structure, and pipeline state tracking.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class SeverityLevel(str, enum.Enum):
    """Clinical severity classification for findings."""
    NORMAL = "NORMAL"
    INCIDENTAL = "INCIDENTAL"
    SIGNIFICANT = "SIGNIFICANT"
    CRITICAL = "CRITICAL"


class ConfidenceLevel(str, enum.Enum):
    """AI confidence classification for model outputs."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AgentStatus(str, enum.Enum):
    """Real-time pipeline agent execution states."""
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class ImageModality(str, enum.Enum):
    """Supported medical imaging modalities."""
    XRAY = "X-RAY"
    MRI = "MRI"
    CT = "CT"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# INPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """Initial client submission containing image and clinical context."""
    image_base64: str = Field(
        ...,
        description="Base64 encoded medical image (PNG/JPG format)"
    )
    symptoms: str = Field(
        default="",
        description="Patient reported symptoms or chief complaint"
    )
    age: Optional[int] = Field(
        default=None, ge=0, le=120, description="Patient age in years"
    )
    sex: Optional[str] = Field(
        default=None, pattern="^(M|F|O)$", description="Patient biological sex"
    )
    clinical_context: Optional[str] = Field(
        default=None, description="Relevant medical history or referral details"
    )

    @field_validator("image_base64")
    @classmethod
    def validate_image_data(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("Invalid or empty base64 image data provided.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# AGENT OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IntakeOutput(BaseModel):
    """Structured data produced by the Intake Agent."""
    validated: bool = Field(default=True, description="Whether input passed validation checks")
    standardized_symptoms: str = Field(default="", description="Clinically normalized symptom description")
    extracted_demographics: Dict[str, Any] = Field(default_factory=dict)
    safety_flags: List[str] = Field(default_factory=list, description="Pre-analysis safety/alert flags")
    recommended_modality: ImageModality = Field(default=ImageModality.UNKNOWN)
    processing_notes: str = Field(default="")


class VisionFinding(BaseModel):
    """Individual anatomical observation from the Vision Agent."""
    anatomical_region: str = Field(..., description="e.g., Left Lung Field, Medial Patella")
    description: str = Field(..., description="Detailed radiological description")
    severity: SeverityLevel = Field(default=SeverityLevel.NORMAL)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.LOW)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    is_anomaly: bool = Field(default=False)


class VisionOutput(BaseModel):
    """Complete visual analysis result from the Vision Agent."""
    modality_detected: ImageModality = Field(default=ImageModality.UNKNOWN)
    technical_quality: str = Field(default="Acceptable", description="Image quality/artifact assessment")
    findings: List[VisionFinding] = Field(default_factory=list)
    overall_assessment: str = Field(default="No obvious abnormalities detected.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeMatch(BaseModel):
    """Differential diagnosis entry from the Research Agent."""
    condition_name: str = Field(..., description="Medical condition or diagnosis")
    match_probability: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage")
    supporting_evidence: str = Field(..., description="Pathophysiological/clinical correlation")
    differential_rank: int = Field(default=0, ge=1)
    icd10_code: Optional[str] = Field(default=None)


class ResearchOutput(BaseModel):
    """Knowledge base search and differential diagnosis result."""
    differential_diagnoses: List[KnowledgeMatch] = Field(default_factory=list)
    matched_conditions: List[str] = Field(default_factory=list)
    relevant_guidelines: List[str] = Field(default_factory=list)
    research_notes: str = Field(default="")
    sources_used: List[str] = Field(default=["internal_knowledge_base"])


# ─────────────────────────────────────────────────────────────────────────────
# REPORT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ReportSection(BaseModel):
    """Standard radiology report structure."""
    clinical_history: str = Field(default="Not provided.")
    technique: str = Field(default="Digital advanced imaging acquisition.")
    findings: str = Field(default="No abnormalities detected.")
    impression: str = Field(default="Within normal limits.")
    recommendations: str = Field(default="Routine follow-up as clinically indicated.")
    disclaimer: str = Field(
        default="This analysis is AI-generated and must be reviewed by a licensed radiologist before any clinical decisions are made."
    )


class FinalReport(BaseModel):
    """Complete synthesized clinical report ready for delivery."""
    report_id: str = Field(default_factory=lambda: f"REP-{uuid.uuid4().hex[:12].upper()}")
    patient_metadata: Dict[str, Any] = Field(default_factory=dict)
    sections: ReportSection = Field(default_factory=ReportSection)
    vision_summary: str = Field(default="")
    research_summary: str = Field(default="")
    overall_severity: SeverityLevel = Field(default=SeverityLevel.NORMAL)
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    agent_pipeline_status: Dict[str, AgentStatus] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT / ADVISOR MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Single turn in the post-report clinical advisor chat."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Incoming question for the ClinicalAdvisorAgent."""
    question: str = Field(..., min_length=3, max_length=1000)


class ChatResponse(BaseModel):
    """Response from the ClinicalAdvisorAgent."""
    answer: str
    report_id: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """Tracks real-time execution state across all agents."""
    current_step: str = Field(default="INIT")
    agent_statuses: Dict[str, AgentStatus] = Field(
        default_factory=lambda: {
            "INTAKE": AgentStatus.WAITING,
            "VISION": AgentStatus.WAITING,
            "RESEARCH": AgentStatus.WAITING,
            "REPORT": AgentStatus.WAITING,
            "CRITIC": AgentStatus.WAITING
        }
    )
    intake_output: Optional[IntakeOutput] = None
    vision_output: Optional[VisionOutput] = None
    research_output: Optional[ResearchOutput] = None
    report_draft: Optional[ReportSection] = None
    final_report: Optional[FinalReport] = None
    error_log: List[str] = Field(default_factory=list)

    def mark_running(self, agent_name: str) -> None:
        self.agent_statuses[agent_name] = AgentStatus.RUNNING

    def mark_done(self, agent_name: str) -> None:
        self.agent_statuses[agent_name] = AgentStatus.DONE

    def mark_error(self, agent_name: str, error_msg: str) -> None:
        self.agent_statuses[agent_name] = AgentStatus.ERROR
        self.error_log.append(f"[{agent_name}] {error_msg}")
