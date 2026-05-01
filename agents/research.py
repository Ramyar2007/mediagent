# mediagent/agents/research.py
"""
Research Agent for MediAgent.
Cross-references vision agent findings against a built-in medical knowledge
base to generate ranked differential diagnoses, ICD-10 mappings, and clinical
correlations. Uses LLM reasoning to weigh evidence and account for demographics.
"""

import logging
from typing import Any, Dict, List, Optional

from core.llm import LLMClient
from core.models import KnowledgeMatch, ResearchOutput, VisionFinding

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN MEDICAL KNOWLEDGE BASE
# Curated set of common radiological findings mapped to clinical conditions.
# Designed for deterministic cross-referencing with LLM reasoning overlay.
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_KB = [
    {
        "condition": "Community-Acquired Pneumonia",
        "icd10": "J18.9",
        "key_findings": ["lobar consolidation", "alveolar opacity", "air bronchograms", "focal infiltrate"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Cardiogenic Pulmonary Edema",
        "icd10": "J81.0",
        "key_findings": ["bilateral perihilar opacities", "kerley B lines", "cephalization", "pleural effusion", "cardiomegaly"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "CRITICAL"
    },
    {
        "condition": "Pleural Effusion",
        "icd10": "J90",
        "key_findings": ["blunting of costophrenic angle", "meniscus sign", "layering fluid", "hemothorax"],
        "modalities": ["X-RAY", "CT", "MRI"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Spontaneous Pneumothorax",
        "icd10": "J93.9",
        "key_findings": ["visceral pleural line", "absence of lung markings", "lung collapse", "hyperlucent hemithorax"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "CRITICAL"
    },
    {
        "condition": "Intracerebral Hemorrhage",
        "icd10": "I61.9",
        "key_findings": ["hyperdense collection", "mass effect", "midline shift", "sulcal effacement", "edema"],
        "modalities": ["CT", "MRI"],
        "typical_severity": "CRITICAL"
    },
    {
        "condition": "Ischemic Stroke",
        "icd10": "I63.9",
        "key_findings": ["hypodensity", "loss of gray-white differentiation", "hypoenhancement", "restricted diffusion"],
        "modalities": ["CT", "MRI"],
        "typical_severity": "CRITICAL"
    },
    {
        "condition": "Intracranial Neoplasm",
        "icd10": "C71.9",
        "key_findings": ["space-occupying lesion", "ring enhancement", "vasogenic edema", "midline shift", "mass effect"],
        "modalities": ["MRI", "CT"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Abdominal Aortic Aneurysm",
        "icd10": "I71.4",
        "key_findings": ["aortic dilation", "circumferential calcification", "thrombus", "rupture signs"],
        "modalities": ["CT", "MRI"],
        "typical_severity": "CRITICAL"
    },
    {
        "condition": "Nephrolithiasis",
        "icd10": "N20.0",
        "key_findings": ["hyperdense calculus", "hydronephrosis", "ureteral dilation", "perinephric stranding"],
        "modalities": ["CT", "X-RAY"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Small Bowel Obstruction",
        "icd10": "K56.6",
        "key_findings": ["dilated loops", "air-fluid levels", "transition point", "collapsed distal bowel"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Long Bone Fracture",
        "icd10": "S82.902",
        "key_findings": ["cortical discontinuity", "displacement", "callus formation", "periosteal reaction", "fracture line"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Degenerative Joint Disease",
        "icd10": "M19.90",
        "key_findings": ["joint space narrowing", "osteophytes", "subchondral sclerosis", "subchondral cysts"],
        "modalities": ["X-RAY", "MRI"],
        "typical_severity": "INCIDENTAL"
    },
    {
        "condition": "Hepatic Steatosis",
        "icd10": "K76.0",
        "key_findings": ["decreased hepatic attenuation", "liver brighter than spleen", "fatty infiltration", "hepatomegaly"],
        "modalities": ["CT", "MRI", "X-RAY"],
        "typical_severity": "INCIDENTAL"
    },
    {
        "condition": "Herniated Disc",
        "icd10": "M51.16",
        "key_findings": ["disc protrusion", "nerve root compression", "thecal sac indentation", "annular tear"],
        "modalities": ["MRI", "CT"],
        "typical_severity": "SIGNIFICANT"
    },
    {
        "condition": "Pulmonary Nodule",
        "icd10": "R91.1",
        "key_findings": ["solitary pulmonary nodule", "ground-glass opacity", "spiculated margins", "calcification pattern"],
        "modalities": ["X-RAY", "CT"],
        "typical_severity": "SIGNIFICANT"
    }
]


class ResearchAgent:
    """
    Knowledge-driven differential diagnosis engine. Matches imaging findings
    to a curated clinical knowledge base, applies demographic weighting, and
    returns ranked diagnostic hypotheses with ICD-10 codes and confidence.
    """

    SYSTEM_PROMPT = """You are a clinical radiology research specialist. Cross-reference imaging findings against the provided knowledge base and return ONLY valid JSON:
{"differential_diagnoses":[{"condition_name":"string","match_probability":0-100,"supporting_evidence":"string","differential_rank":1,"icd10_code":"string"}],"matched_conditions":["string"],"relevant_guidelines":["string"],"research_notes":"string"}

Rules:
1. ONLY use conditions from the provided KB. Do not invent diagnoses.
2. Match anatomical regions and radiological descriptors to KB key_findings.
3. Factor in demographics (age, sex, comorbidities) to adjust probabilities.
4. Output 2-4 differentials maximum, ranked highest to lowest probability.
5. Use exact ICD-10 codes from the KB.
6. Skip conditions with no imaging evidence. Never force-fit.
7. Minimum probability 5%. Never output 0%.
8. Each supporting_evidence must explain WHY the condition matches (one full sentence minimum).
9. No markdown, no commentary — JSON only."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def process(self, vision_findings: List[VisionFinding], demographics: Dict[str, Any] = None, detected_modality: str = "UNKNOWN") -> ResearchOutput:
        """
        Execute knowledge-base cross-referencing and differential generation.
        
        Args:
            vision_findings: List of structured findings from Vision Agent
            demographics: Patient metadata from Intake Agent
            
        Returns:
            ResearchOutput: Ranked differentials, matched conditions, and clinical notes
        """
        logger.info("🔍 Research Agent initiated differential diagnosis matching")

        demographics = demographics or {}
        findings_text = self._format_findings_for_prompt(vision_findings)
        kb_text = self._format_kb_for_prompt(detected_modality)

        user_prompt = f"""Patient Demographics:
- Age: {demographics.get('age', 'Unknown')}
- Sex: {demographics.get('sex', 'Unknown')}
- Comorbidities: {demographics.get('comorbidities', 'None reported')}

Vision Agent Findings:
{findings_text}

Medical Knowledge Base:
{kb_text}

Analyze the findings, match them against the knowledge base, factor in demographics, and return the ranked differential diagnosis in the specified JSON format."""

        result = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            force_json=True
        )

        if not result.get("success"):
            logger.error(f"❌ Research LLM call failed: {result.get('error')}")
            return self._get_fallback_output()

        raw_content = result.get("content", "")
        parsed = LLMClient.extract_json_from_response(raw_content)
        if not parsed:
            logger.warning("⚠️ Failed to parse research LLM JSON response. Using fallback.")
            return self._get_fallback_output()

        try:
            return self._parse_research_response(parsed)
        except Exception as e:
            logger.error(f"💥 Research response mapping failed: {e}")
            return self._get_fallback_output()

    def _format_findings_for_prompt(self, findings: List[VisionFinding]) -> str:
        """Convert VisionFinding objects into LLM-readable text blocks."""
        if not findings:
            return "No specific findings reported by vision agent. Image appears unremarkable."
        blocks = []
        for i, f in enumerate(findings, 1):
            blocks.append(
                f"[{i}] Region: {f.anatomical_region} | "
                f"Description: {f.description} | "
                f"Severity: {f.severity.value} | "
                f"Confidence: {f.confidence.value} ({f.confidence_score:.1f}%) | "
                f"Anomaly: {'Yes' if f.is_anomaly else 'No'}"
            )
        return "\n".join(blocks)

    def _format_kb_for_prompt(self, modality: str = "UNKNOWN") -> str:
        """Format the KB into a structured reference block, pre-filtered by modality."""
        # Filter to only conditions compatible with the detected modality
        if modality in ("X-RAY", "CT", "MRI"):
            relevant = [e for e in MEDICAL_KB if modality in e["modalities"]]
        else:
            relevant = MEDICAL_KB

        lines = ["[CONDITION REFERENCE TABLE]"]
        for entry in relevant:
            lines.append(
                f"- {entry['condition']} (ICD-10: {entry['icd10']}) | "
                f"Findings: {', '.join(entry['key_findings'])} | "
                f"Severity: {entry['typical_severity']}"
            )
        return "\n".join(lines)

    def _parse_research_response(self, data: Dict[str, Any]) -> ResearchOutput:
        """Validate and map LLM output to ResearchOutput model."""
        raw_diffs = data.get("differential_diagnoses", [])
        differentials = []

        for rank, item in enumerate(raw_diffs, 1):
            try:
                match = KnowledgeMatch(
                    condition_name=str(item.get("condition_name", "Unknown Condition")),
                    match_probability=float(item.get("match_probability", 0.0)),
                    supporting_evidence=str(item.get("supporting_evidence", "Insufficient data for correlation.")),
                    differential_rank=rank,
                    icd10_code=str(item.get("icd10_code", "Z00.00"))
                )
                differentials.append(match)
            except Exception as e:
                logger.warning(f"⚠️ Skipping malformed differential entry: {e}")
                continue

        matched_conditions = [d.condition_name for d in differentials]
        guidelines = data.get("relevant_guidelines", ["ACR Appropriateness Criteria", "NICE Imaging Guidelines"])
        notes = data.get("research_notes", "Standard knowledge-base cross-referencing applied.")

        return ResearchOutput(
            differential_diagnoses=differentials,
            matched_conditions=matched_conditions,
            relevant_guidelines=guidelines,
            research_notes=notes,
            sources_used=["internal_knowledge_base", "ac_radiology_standards"]
        )

    def _get_fallback_output(self) -> ResearchOutput:
        """Safe fallback when KB matching fails."""
        logger.warning("⚠️ Returning fallback ResearchOutput.")
        return ResearchOutput(
            differential_diagnoses=[],
            matched_conditions=[],
            relevant_guidelines=["Manual radiologist review required"],
            research_notes="Knowledge base matching failed. Clinical correlation strongly recommended.",
            sources_used=["internal_knowledge_base"]
        )
