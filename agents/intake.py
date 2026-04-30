# mediagent/agents/intake.py
"""
Intake Agent for MediAgent.
Validates patient submissions, normalizes clinical terminology,
extracts demographics, detects imaging modality hints, and flags
urgent safety concerns before routing to downstream agents.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from core.llm import LLMClient
from core.models import IntakeOutput, ImageModality, PatientInput

logger = logging.getLogger(__name__)


class IntakeAgent:
    """
    First-stage pipeline agent responsible for input validation,
    clinical text normalization, demographic extraction, and safety triage.
    Ensures downstream agents receive structured, standardized input.
    """

    # Deterministic safety keywords for immediate flagging
    SAFETY_KEYWORDS = [
        "acute trauma", "chest pain", "shortness of breath", "dyspnea",
        "stroke symptoms", "neurological deficit", "hemoptysis", "massive bleed",
        "pediatric emergency", "pregnant", "anaphylaxis", "sepsis", "fever",
        "head injury", "spinal trauma", "acute abdomen", "suspected fracture"
    ]

    MODALITY_KEYWORDS = {
        "x-ray": ImageModality.XRAY,
        "xr": ImageModality.XRAY,
        "radiograph": ImageModality.XRAY,
        "ct scan": ImageModality.CT,
        "ct": ImageModality.CT,
        "computed tomography": ImageModality.CT,
        "mri": ImageModality.MRI,
        "magnetic resonance": ImageModality.MRI,
        "mammogram": ImageModality.XRAY,  # Technically X-ray based
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def process(self, patient_input: PatientInput) -> IntakeOutput:
        """
        Main intake processing method.
        
        Args:
            patient_input: Raw validated patient submission
            
        Returns:
            IntakeOutput: Structured, normalized, safety-checked data
        """
        logger.info("📋 Intake Agent processing initiated")
        
        # 1. Validate image payload
        if not self._validate_image_payload(patient_input.image_base64):
            logger.warning("⚠️ Image payload validation failed. Proceeding with warnings.")
            
        # 2. Apply deterministic safety triage
        safety_flags = self._check_deterministic_safety(patient_input)
        
        # 3. Clinical normalization & demographic extraction via LLM
        structured_data = self._normalize_with_llm(patient_input, safety_flags)
        
        # 4. Enrich modality detection
        modality = self._infer_modality(patient_input, structured_data)
        
        # 5. Assemble & validate output
        try:
            output = IntakeOutput(
                validated=True,
                standardized_symptoms=structured_data.get("standardized_symptoms", patient_input.symptoms or ""),
                extracted_demographics=structured_data.get("extracted_demographics", {}),
                safety_flags=list(set(safety_flags + structured_data.get("safety_flags", []))),
                recommended_modality=modality,
                processing_notes=structured_data.get("processing_notes", "")
            )
            logger.info("✅ Intake Agent completed successfully")
            return output
        except Exception as e:
            logger.error(f"💥 IntakeOutput validation failed: {e}")
            return self._get_fallback_output(patient_input, safety_flags)

    def _validate_image_payload(self, base64_data: str) -> bool:
        """Validate base64 image integrity and size constraints."""
        if not base64_data or len(base64_data) < 500:
            return False
        # Check for valid base64 pattern (ignoring data URI prefix)
        clean = re.sub(r"^data:image/[a-z]+;base64,", "", base64_data)
        try:
            import base64
            base64.b64decode(clean)
            return len(clean) < 20 * 1024 * 1024  # < 20MB limit
        except Exception:
            return False

    def _check_deterministic_safety(self, inp: PatientInput) -> List[str]:
        """Scan raw input for high-priority clinical safety terms."""
        text = f"{inp.symptoms} {inp.clinical_context}".lower()
        flags = []
        for kw in self.SAFETY_KEYWORDS:
            if kw.lower() in text:
                flags.append(f"URGENT_TERM_DETECTED: {kw}")
        if inp.age is not None and inp.age < 18:
            flags.append("PATIENT_AGE: PEDIATRIC_REQUIRES_EXPERT_REVIEW")
        if inp.age is not None and inp.age > 75:
            flags.append("PATIENT_AGE: GERIATRIC_CONSIDERATIONS_RECOMMENDED")
        return flags

    def _normalize_with_llm(self, inp: PatientInput, existing_flags: List[str]) -> Dict[str, Any]:
        """Use LLM to standardize clinical text and extract structured demographics."""
        prompt = f"""You are a clinical data standardization expert. 
Convert the following raw patient input into standardized clinical terminology and extract demographics.
Respond ONLY with a JSON object matching this schema:
{{
  "standardized_symptoms": "string",
  "extracted_demographics": {{ "age": int|null, "sex": "M|F|O"|null, "comorbidities": ["string"] }},
  "safety_flags": ["string"],
  "processing_notes": "string"
}}

Raw Input:
- Symptoms: "{inp.symptoms or 'Not provided'}"
- Age: {inp.age}
- Sex: {inp.sex}
- Clinical Context: "{inp.clinical_context or 'Not provided'}"
- Existing Safety Flags: {existing_flags}

Guidelines:
1. Convert layman symptoms to medical terminology (e.g., "can't breathe" -> "dyspnea", "stomach pain" -> "abdominal pain").
2. Extract or infer comorbidities from context if mentioned.
3. Add any additional clinical safety flags not already detected.
4. Provide concise processing notes summarizing data quality.
5. Maintain exact JSON structure. No markdown, no extra text."""

        result = self.llm.generate_text(prompt=prompt, force_json=True)
        if result.get("success") and result.get("content"):
            parsed = LLMClient.extract_json_from_response(result["content"])
            if parsed:
                return parsed
        
        logger.warning("⚠️ LLM normalization failed. Using deterministic fallback.")
        return self._build_fallback_dict(inp, existing_flags)

    def _infer_modality(self, inp: PatientInput, llm_data: Dict[str, Any]) -> ImageModality:
        """Infer imaging modality from text hints or default to UNKNOWN."""
        text = f"{inp.symptoms} {inp.clinical_context}".lower()
        for kw, mod in self.MODALITY_KEYWORDS.items():
            if kw in text:
                return mod
        return ImageModality.UNKNOWN

    def _build_fallback_dict(self, inp: PatientInput, flags: List[str]) -> Dict[str, Any]:
        """Deterministic fallback when LLM is unavailable."""
        return {
            "standardized_symptoms": inp.symptoms or "No symptoms provided",
            "extracted_demographics": {
                "age": inp.age,
                "sex": inp.sex,
                "comorbidities": []
            },
            "safety_flags": flags,
            "processing_notes": "LLM normalization unavailable. Raw input preserved."
        }

    def _get_fallback_output(self, inp: PatientInput, flags: List[str]) -> IntakeOutput:
        """Return a safe, minimally structured IntakeOutput on critical failure."""
        return IntakeOutput(
            validated=False,
            standardized_symptoms=inp.symptoms or "",
            extracted_demographics={"age": inp.age, "sex": inp.sex, "comorbidities": []},
            safety_flags=flags + ["INTAKE_AGENT_FALLBACK_MODE"],
            recommended_modality=ImageModality.UNKNOWN,
            processing_notes="Intake agent encountered critical validation failure. Pipeline continues with degraded state."
        )
