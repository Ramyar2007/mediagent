# mediagent/core/fhir.py
"""
FHIR R4 DiagnosticReport export for MediAgent.
Converts a FinalReport into a standards-compliant HL7 FHIR R4 resource
suitable for import into any EMR system (Epic, Cerner, etc.).
"""

import base64
import uuid
from datetime import datetime
from typing import Any, Dict, List

from core.models import FinalReport


def to_fhir_diagnostic_report(report: FinalReport) -> Dict[str, Any]:
    """
    Convert a MediAgent FinalReport into a FHIR R4 DiagnosticReport resource.

    Conforms to:
      http://hl7.org/fhir/R4/diagnosticreport.html
      LOINC 18748-4 (Diagnostic imaging study)
    """
    meta = report.patient_metadata or {}
    sections = report.sections
    ts = report.generation_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    patient_uid = f"patient-{uuid.uuid4().hex[:8]}"

    # ── Contained Patient resource ────────────────────────────────────────────
    patient: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_uid,
    }
    if meta.get("sex"):
        patient["gender"] = _map_sex(str(meta["sex"]))
    age = meta.get("age")
    if age:
        birth_year = datetime.now().year - int(age)
        patient["birthDate"] = str(birth_year)
    if meta.get("patient_name"):
        patient["name"] = [{"text": str(meta["patient_name"])}]

    # ── Imaging study Observations (one per finding section) ─────────────────
    observations: List[Dict[str, Any]] = []
    finding_text = sections.findings or ""
    if finding_text:
        obs_id = f"obs-{uuid.uuid4().hex[:8]}"
        observations.append({
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "59776-5",
                    "display": "Findings"
                }]
            },
            "subject": {"reference": f"#{patient_uid}"},
            "effectiveDateTime": ts,
            "valueString": finding_text[:2000]  # FHIR string limit safeguard
        })

    # ── Build contained resources list ───────────────────────────────────────
    contained: List[Dict] = [patient] + observations
    obs_refs = [{"reference": f"#{o['id']}"} for o in observations]

    # ── Severity → SNOMED code ────────────────────────────────────────────────
    severity_val = report.overall_severity.value if hasattr(report.overall_severity, "value") else str(report.overall_severity)
    conclusion_codes = _severity_to_snomed(severity_val)

    # ── Modality coding from vision summary ──────────────────────────────────
    modality_code = _extract_modality_code(report.vision_summary)

    # ── Presented form: full plain-text report ────────────────────────────────
    report_text = (
        f"CLINICAL HISTORY\n{sections.clinical_history}\n\n"
        f"TECHNIQUE\n{sections.technique}\n\n"
        f"FINDINGS\n{sections.findings}\n\n"
        f"IMPRESSION\n{sections.impression}\n\n"
        f"RECOMMENDATIONS\n{sections.recommendations}\n\n"
        f"DISCLAIMER\n{sections.disclaimer}"
    )

    # ── Agent pipeline extension ──────────────────────────────────────────────
    pipeline_str = ", ".join(
        f"{k}:{v.value}" for k, v in report.agent_pipeline_status.items()
    )

    resource: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": report.report_id,
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/DiagnosticReport"
            ],
            "lastUpdated": ts,
            "tag": [{
                "system": "https://mediagent.ai/tags",
                "code": "ai-generated",
                "display": "AI Generated — Requires Radiologist Review"
            }]
        },
        "contained": contained,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "RAD",
                "display": "Radiology"
            }]
        }],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                },
                modality_code
            ],
            "text": "Medical Imaging Analysis — AI-Assisted Radiology Report"
        },
        "subject": {
            "reference": f"#{patient_uid}"
        },
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{
            "display": "MediAgent AI System (AMD Instinct MI300X)"
        }],
        "result": obs_refs,
        "conclusion": sections.impression,
        "conclusionCode": conclusion_codes,
        "presentedForm": [{
            "contentType": "text/plain; charset=utf-8",
            "data": base64.b64encode(report_text.encode("utf-8")).decode("utf-8"),
            "title": f"Radiology Report {report.report_id}",
            "creation": ts
        }],
        "extension": [
            {
                "url": "https://mediagent.ai/fhir/StructureDefinition/ai-quality-score",
                "valueInteger": _extract_qa_score(sections.recommendations)
            },
            {
                "url": "https://mediagent.ai/fhir/StructureDefinition/overall-severity",
                "valueCode": severity_val
            },
            {
                "url": "https://mediagent.ai/fhir/StructureDefinition/pipeline-status",
                "valueString": pipeline_str
            },
            {
                "url": "https://mediagent.ai/fhir/StructureDefinition/inference-platform",
                "valueString": "AMD Instinct MI300X / ROCm / vLLM / Qwen"
            }
        ]
    }

    return resource


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_sex(sex: str) -> str:
    return {"M": "male", "F": "female", "O": "other"}.get(sex.upper(), "unknown")


def _severity_to_snomed(severity: str) -> List[Dict]:
    snomed = {
        "NORMAL":      ("260313008", "Normal"),
        "INCIDENTAL":  ("102483000", "Incidental finding"),
        "SIGNIFICANT": ("404684003", "Clinical finding"),
        "CRITICAL":    ("399625000", "Critical finding"),
    }
    code, display = snomed.get(severity.upper(), ("404684003", "Clinical finding"))
    return [{
        "coding": [{
            "system": "http://snomed.info/sct",
            "code": code,
            "display": display
        }]
    }]


def _extract_modality_code(vision_summary: str) -> Dict:
    """Map detected modality to DICOM/RadLex LOINC codes."""
    loinc_map = {
        "X-RAY": ("24627-2", "Chest X-ray"),
        "CT":    ("18747-6", "CT study"),
        "MRI":   ("18755-9", "MRI study"),
    }
    for key, (code, display) in loinc_map.items():
        if key in vision_summary.upper():
            return {"system": "http://loinc.org", "code": code, "display": display}
    return {"system": "http://loinc.org", "code": "18748-4", "display": "Diagnostic imaging study"}


def _extract_qa_score(recommendations: str) -> int:
    import re
    m = re.search(r"Score[:\s]+(\d+)", recommendations or "")
    return int(m.group(1)) if m else 85
