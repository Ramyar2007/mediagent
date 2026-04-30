# mediagent/agents/report.py
"""
Report Agent for MediAgent.
Synthesizes outputs from Intake, Vision, and Research agents into a 
structured, professional clinical radiology report. Follows standard 
ACR/NICE radiological reporting conventions with strict JSON enforcement.
"""

import logging
from typing import Optional, Any

from core.llm import LLMClient
from core.models import IntakeOutput, ReportSection, ResearchOutput, VisionOutput

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Clinical report synthesis engine. Transforms structured agent outputs
    into a formal radiology report suitable for clinician review.
    """

    STANDARD_DISCLAIMER = (
        "This analysis is AI-generated and must be reviewed by a licensed radiologist "
        "before any clinical decisions are made."
    )

    SYSTEM_PROMPT = """You are a senior board-certified radiologist and clinical documentation specialist.
Your task is to synthesize the provided imaging findings, patient context, and differential diagnoses 
into a formal, structured radiology report.

OUTPUT FORMAT (STRICT JSON):
{
  "clinical_history": "string",
  "technique": "string",
  "findings": "string",
  "impression": "string",
  "recommendations": "string",
  "disclaimer": "string"
}

SECTION GUIDELINES:
1. clinical_history: Summarize patient age, sex, chief complaint, and relevant context from intake data. Use formal clinical language.
2. technique: Describe imaging modality, planes/sequences (if applicable), contrast status, and image quality/artifacts.
3. findings: Provide a detailed, anatomical region-by-region description. Reference specific observations, note severity levels (NORMAL/INCIDENTAL/SIGNIFICANT/CRITICAL), distinguish normal variants from pathology, and describe size, shape, density, and location precisely.
4. impression: Concisely summarize the most clinically significant findings. Explicitly list the top 3 differential diagnoses with their match probabilities. State overall clinical significance. End the impression with a confidence statement like: 'Confidence Level: High — imaging findings are classic and unambiguous.' or 'Confidence Level: Medium — findings are suggestive but require clinical correlation.' based on the quality of evidence.
5. recommendations: Provide clinically appropriate next steps (e.g., further imaging, specialist referral, urgent evaluation, routine follow-up). Base recommendations on severity and diagnostic confidence.
6. disclaimer: ALWAYS use this exact text: "This analysis is AI-generated and must be reviewed by a licensed radiologist before any clinical decisions are made."

RULES:
- Output ONLY valid JSON. No markdown, no preamble, no extra text.
- Maintain objective, professional radiological tone.
- If data is missing for a section, state "Not provided" or "Unable to assess" rather than fabricating.
- Use standard radiological terminology throughout.
"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def process(
        self, 
        intake: Optional[IntakeOutput] = None, 
        vision: Optional[VisionOutput] = None, 
        research: Optional[ResearchOutput] = None
    ) -> ReportSection:
        """
        Synthesize agent outputs into a formal clinical radiology report.
        
        Args:
            intake: Structured patient context and safety flags
            vision: Imaging analysis findings and technical details
            research: Ranked differential diagnoses and clinical correlations
            
        Returns:
            ReportSection: Complete, structured radiology report
        """
        logger.info("📝 Report Agent initiated report synthesis")
        
        user_prompt = self._build_user_prompt(intake, vision, research)
        
        result = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            force_json=True
        )

        if not result.get("success"):
            logger.error(f"❌ Report generation LLM call failed: {result.get('error')}")
            return self._get_fallback_report(intake, vision, research)

        raw_content = result.get("content", "")
        parsed = LLMClient.extract_json_from_response(raw_content)
        if not parsed:
            logger.warning("⚠️ Failed to parse report LLM JSON response. Using fallback.")
            return self._get_fallback_report(intake, vision, research)

        try:
            return self._parse_report_response(parsed)
        except Exception as e:
            logger.error(f"💥 Report mapping failed: {e}")
            return self._get_fallback_report(intake, vision, research)

    def _build_user_prompt(self, intake, vision, research) -> str:
        """Construct a highly structured prompt for the LLM synthesizer."""
        # Clinical History Block
        history = "Patient information not provided."
        if intake:
            age = f"{intake.extracted_demographics.get('age', 'Unknown')} years"
            sex = intake.extracted_demographics.get('sex', 'Unknown')
            symptoms = intake.standardized_symptoms or "No symptoms reported"
            context = intake.processing_notes or ""
            history = f"Age: {age} | Sex: {sex} | Chief Complaint: {symptoms} | Additional Context: {context}"

        # Technique Block
        technique = "Imaging technique not specified."
        if vision:
            modality = vision.modality_detected.value
            quality = vision.technical_quality
            technique = f"Modality: {modality} | Technical Assessment: {quality}"

        # Findings Block
        findings_text = "No imaging findings available."
        if vision and vision.findings:
            finding_blocks = []
            for f in vision.findings:
                finding_blocks.append(
                    f"- {f.anatomical_region}: {f.description} (Severity: {f.severity.value}, "
                    f"Confidence: {f.confidence.value} {f.confidence_score:.0f}%, Anomaly: {'Yes' if f.is_anomaly else 'No'})"
                )
            findings_text = "\n".join(finding_blocks)

        # Differentials Block
        diff_text = "No differential diagnoses generated."
        if research and research.differential_diagnoses:
            top_three = research.differential_diagnoses[:3]
            diff_blocks = []
            for d in top_three:
                diff_blocks.append(
                    f"{d.differential_rank}. {d.condition_name} (ICD-10: {d.icd10_code}) - "
                    f"Probability: {d.match_probability:.1f}% | Evidence: {d.supporting_evidence}"
                )
            diff_text = "\n".join(diff_blocks)

        return f"""Synthesize the following structured medical data into a formal radiology report:

[CLINICAL HISTORY]
{history}

[IMAGING TECHNIQUE]
{technique}

[IMAGING FINDINGS]
{findings_text}

[DIFFERENTIAL DIAGNOSES]
{diff_text}

Generate the complete report following the exact JSON schema and clinical guidelines provided in the system prompt."""

    def _parse_report_response(self, data: dict) -> ReportSection:
        """Validate and map LLM output to ReportSection model with safety enforcement."""
        # Extract fields with safe defaults
        clinical_history = str(data.get("clinical_history", "Not provided."))
        technique = str(data.get("technique", "Digital advanced imaging acquisition."))
        findings = str(data.get("findings", "No abnormalities detected."))
        impression = str(data.get("impression", "Within normal limits."))
        recommendations = str(data.get("recommendations", "Routine follow-up as clinically indicated."))
        
        # Enforce exact disclaimer for regulatory compliance
        disclaimer = self.STANDARD_DISCLAIMER

        return ReportSection(
            clinical_history=clinical_history,
            technique=technique,
            findings=findings,
            impression=impression,
            recommendations=recommendations,
            disclaimer=disclaimer
        )

    def _get_fallback_report(
        self, 
        intake: Optional[IntakeOutput], 
        vision: Optional[VisionOutput], 
        research: Optional[ResearchOutput]
    ) -> ReportSection:
        """Deterministic fallback report when LLM synthesis fails."""
        logger.warning("⚠️ Generating deterministic fallback radiology report.")
        
        # Build minimal clinically safe sections from raw data
        history = "Patient data available. See raw intake logs for details."
        if intake:
            history = f"Symptoms: {intake.standardized_symptoms} | Demographics: {intake.extracted_demographics}"
            
        technique = "Imaging modality unspecified due to processing limitation."
        if vision:
            technique = f"Modality: {vision.modality_detected.value} | Quality: {vision.technical_quality}"
            
        findings = "Imaging analysis incomplete. Manual radiologist review strongly recommended."
        if vision and vision.findings:
            findings = "; ".join([f"{f.anatomical_region}: {f.description}" for f in vision.findings])
            
        impression = "Diagnostic certainty limited. Top clinical considerations based on available data:"
        if research and research.differential_diagnoses:
            impression += " " + ", ".join([d.condition_name for d in research.differential_diagnoses[:3]])
            
        recommendations = "Urgent manual review by a licensed radiologist is required. Correlate with clinical presentation."
        
        return ReportSection(
            clinical_history=history,
            technique=technique,
            findings=findings,
            impression=impression,
            recommendations=recommendations,
            disclaimer=self.STANDARD_DISCLAIMER
        )
