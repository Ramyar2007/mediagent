# mediagent/agents/vision.py
"""
Vision Agent for MediAgent.
Multimodal medical image analysis engine that processes base64-encoded
diagnostic images using local Qwen vision capabilities. Extracts
anatomical structures, detects pathologies, and structures findings
into standardized radiological observations with confidence scoring.
"""

import logging
from typing import Any, Dict, List, Optional

from core.llm import LLMClient
from core.models import (
    ConfidenceLevel,
    ImageModality,
    SeverityLevel,
    VisionFinding,
    VisionOutput,
)

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Specialized radiological analysis agent. Interprets medical imagery
    (X-ray, MRI, CT) and outputs structured findings with severity and
    confidence classifications. Operates with deterministic fallbacks
    to maintain pipeline continuity on LLM failures.
    """

    SYSTEM_PROMPT = """You are a board-certified radiologist with expertise in multimodal medical imaging analysis.
Your task is to analyze the provided medical image and produce a strictly structured JSON assessment.

Output Schema:
{
  "modality_detected": "X-RAY" | "MRI" | "CT" | "UNKNOWN",
  "technical_quality": "string describing image quality, exposure, artifacts, positioning, and clinical adequacy",
  "findings": [
    {
      "anatomical_region": "string (e.g., Right Upper Lung Field, Cerebral Ventricles, Left Femoral Neck)",
      "description": "string (detailed radiological observation using standard anatomical and pathological terminology)",
      "severity": "NORMAL" | "INCIDENTAL" | "SIGNIFICANT" | "CRITICAL",
      "confidence": "LOW" | "MEDIUM" | "HIGH",
      "confidence_score": number (0.0 to 100.0),
      "is_anomaly": boolean
    }
  ],
  "overall_assessment": "string (concise, clinically relevant summary of the imaging study)"
}

Clinical Guidelines:
1. Use precise radiological terminology (e.g., opacity, lucency, effacement, enhancement, calcification, edema, mass effect).
2. Reference anatomical orientation correctly (AP/PA, axial/coronal/sagittal, left/right, medial/lateral, proximal/distal).
3. Differentiate clearly between normal anatomical variants, degenerative changes, and acute/pathological findings.
4. Assign confidence scores realistically based on image resolution, contrast, artifacts, and visibility of the region.
5. If the image is non-diagnostic or severely artifacted, note this in technical_quality and set overall_assessment accordingly.
6. Do NOT provide treatment plans, medication recommendations, or definitive clinical diagnoses beyond imaging observations.
7. Output ONLY valid JSON. No markdown formatting, no preamble, no postscript.
"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def process(self, image_base64: str, clinical_context: str = "") -> VisionOutput:
        """
        Execute multimodal image analysis.
        
        Args:
            image_base64: Base64 encoded medical image
            clinical_context: Standardized symptoms/demographics from Intake Agent
            
        Returns:
            VisionOutput: Structured radiological findings and metadata
        """
        logger.info("👁️ Vision Agent initiated multimodal analysis")

        user_prompt = "Analyze this medical image carefully."
        if clinical_context:
            user_prompt += f"\n\nClinical Context: {clinical_context}"
        user_prompt += "\n\nProvide a structured radiological assessment following the specified JSON schema."

        result = self.llm.generate_vision(
            base64_image=image_base64,
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=4096
        )

        if not result.get("success"):
            logger.error(f"❌ Vision LLM call failed: {result.get('error')}")
            return self._get_fallback_output()

        raw_content = result.get("content", "")
        parsed = LLMClient.extract_json_from_response(raw_content)
        if not parsed:
            logger.warning("⚠️ Failed to parse vision LLM JSON response. Using fallback.")
            return self._get_fallback_output()

        try:
            return self._parse_vision_response(parsed, result.get("usage", {}))
        except Exception as e:
            logger.error(f"💥 Vision response mapping failed: {e}")
            return self._get_fallback_output()

    def _parse_vision_response(self, data: Dict[str, Any], usage: Dict[str, int]) -> VisionOutput:
        """Map raw LLM JSON to validated Pydantic models with safe enum conversion."""
        findings = []
        raw_findings = data.get("findings", [])

        for item in raw_findings:
            try:
                finding = VisionFinding(
                    anatomical_region=item.get("anatomical_region", "Unspecified Region"),
                    description=item.get("description", "Unable to generate detailed description."),
                    severity=self._safe_enum(SeverityLevel, item.get("severity"), SeverityLevel.NORMAL),
                    confidence=self._safe_enum(ConfidenceLevel, item.get("confidence"), ConfidenceLevel.MEDIUM),
                    confidence_score=float(item.get("confidence_score", 50.0)),
                    is_anomaly=bool(item.get("is_anomaly", False))
                )
                findings.append(finding)
            except Exception as e:
                logger.warning(f"⚠️ Skipping malformed finding due to validation error: {e}")
                continue

        modality_str = data.get("modality_detected", "UNKNOWN").upper()
        try:
            modality = ImageModality(modality_str)
        except ValueError:
            modality = ImageModality.UNKNOWN

        return VisionOutput(
            modality_detected=modality,
            technical_quality=data.get(
                "technical_quality", 
                "Image quality acceptable for preliminary assessment."
            ),
            findings=findings,
            overall_assessment=data.get(
                "overall_assessment", 
                "Unable to generate overall assessment from provided data."
            ),
            metadata={
                "llm_usage": usage,
                "findings_count": len(findings),
                "anomalies_detected": sum(1 for f in findings if f.is_anomaly)
            }
        )

    @staticmethod
    def _safe_enum(enum_cls, value, default):
        """Safely convert string to enum, falling back gracefully."""
        try:
            return enum_cls(str(value).strip().upper())
        except (ValueError, AttributeError, TypeError):
            return default

    def _get_fallback_output(self) -> VisionOutput:
        """Return a safe, non-failure-breaking VisionOutput when processing fails."""
        logger.warning("⚠️ Returning fallback VisionOutput due to processing failure.")
        return VisionOutput(
            modality_detected=ImageModality.UNKNOWN,
            technical_quality="Analysis unavailable due to system error. Manual review required.",
            findings=[],
            overall_assessment="Vision analysis could not be completed. Please verify image quality and system connectivity.",
            metadata={"error": "VISION_AGENT_FALLBACK", "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        )
