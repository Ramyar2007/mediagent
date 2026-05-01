# mediagent/agents/critic.py
"""
Critic Agent for MediAgent.
Final quality control and peer-review layer. Validates report consistency,
flags low-confidence observations, enforces regulatory disclaimers, assigns
quality scores, and applies corrective refinements before final delivery.
"""

import logging
from typing import Any, Dict, List, Optional

from core.llm import LLMClient
from core.models import AgentStatus, PipelineState, ReportSection, VisionOutput

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Medical QA/QC engine. Cross-validates the synthesized report against
    upstream agent outputs, detects logical inconsistencies, enforces
    clinical safety thresholds, and produces a finalized, auditable report.
    """

    STANDARD_DISCLAIMER = (
        "This analysis is AI-generated and must be reviewed by a licensed radiologist "
        "before any clinical decisions are made."
    )

    SYSTEM_PROMPT = """You are a senior radiology peer-reviewer. Evaluate the draft report against the vision findings and return ONLY valid JSON:
{"clinical_history":"string","technique":"string","findings":"string","impression":"string","recommendations":"string","disclaimer":"string","quality_score":0-100,"review_issues":["string"],"uncertainty_warnings":["string"]}

Review criteria:
1. CONSISTENCY: Every vision anomaly must appear in the report. Flag contradictions.
2. CONFIDENCE: Findings < 50% confidence or LOW → add uncertainty warning, recommend confirmatory imaging.
3. DISCLAIMER: Must be exactly: "This analysis is AI-generated and must be reviewed by a licensed radiologist before any clinical decisions are made."
4. TONE: Formal, objective radiological language. No speculative phrasing or definitive claims without imaging evidence.
5. QUALITY SCORE: completeness 30% + accuracy/consistency 40% + clinical safety 20% + compliance 10%.
6. Apply corrections directly. No placeholders or TODOs.
No markdown. Never fabricate findings. Lower score for failed pipeline agents."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.last_quality_score: int = 100
        self.last_review_issues: List[str] = []
        self.last_uncertainty_warnings: List[str] = []

    def process(self, draft_report: ReportSection, pipeline_state: PipelineState) -> ReportSection:
        """
        Execute final peer review and quality enforcement.
        
        Args:
            draft_report: Unreviewed ReportSection from Report Agent
            pipeline_state: Full execution context including vision findings and agent statuses
            
        Returns:
            ReportSection: Finalized, QA-reviewed clinical report
        """
        logger.info("🛡️ Critic Agent initiated final quality review")
        
        user_prompt = self._build_review_prompt(draft_report, pipeline_state)
        
        result = self.llm.generate_text(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            force_json=True
        )

        if not result.get("success"):
            logger.error(f"❌ Critic LLM call failed: {result.get('error')}")
            return self._apply_deterministic_qa(draft_report, pipeline_state)

        raw_content = result.get("content", "")
        parsed = LLMClient.extract_json_from_response(raw_content)
        if not parsed:
            logger.warning("⚠️ Failed to parse critic JSON response. Applying deterministic QA.")
            return self._apply_deterministic_qa(draft_report, pipeline_state)

        try:
            return self._parse_qa_response(parsed, pipeline_state)
        except Exception as e:
            logger.error(f"💥 Critic mapping failed: {e}")
            return self._apply_deterministic_qa(draft_report, pipeline_state)

    def _build_review_prompt(self, draft: ReportSection, state: PipelineState) -> str:
        """Format pipeline context and draft report for LLM critique."""
        # Extract vision findings for cross-reference
        vision_text = "No vision findings available."
        if state.vision_output:
            findings_list = []
            for f in state.vision_output.findings:
                findings_list.append(
                    f"- Region: {f.anatomical_region} | Desc: {f.description} | "
                    f"Severity: {f.severity.value} | Confidence: {f.confidence.value} ({f.confidence_score}%) | "
                    f"Anomaly: {f.is_anomaly}"
                )
            vision_text = "\n".join(findings_list) if findings_list else "No specific findings."

        # Agent execution status
        agent_status = ", ".join([f"{k}: {v.value}" for k, v in state.agent_statuses.items()])

        return f"""[PIPELINE EXECUTION STATUS]
{agent_status}

[VISION AGENT FINDINGS FOR CROSS-REFERENCE]
{vision_text}

[DRAFT REPORT FOR REVIEW]
Clinical History: {draft.clinical_history}
Technique: {draft.technique}
Findings: {draft.findings}
Impression: {draft.impression}
Recommendations: {draft.recommendations}
Disclaimer: {draft.disclaimer}

Critique the draft against the vision findings and pipeline status. Apply corrections, flag uncertainties, verify compliance, and output the refined JSON report."""

    def _parse_qa_response(self, data: Dict[str, Any], state: PipelineState) -> ReportSection:
        """Validate, extract, and enforce QA metadata on the report."""
        # Extract core report fields
        draft = ReportSection(
            clinical_history=str(data.get("clinical_history", "Not provided.")),
            technique=str(data.get("technique", "Imaging technique not specified.")),
            findings=str(data.get("findings", "No abnormalities detected.")),
            impression=str(data.get("impression", "Within normal limits.")),
            recommendations=str(data.get("recommendations", "Routine follow-up as clinically indicated.")),
            disclaimer=str(data.get("disclaimer", self.STANDARD_DISCLAIMER))
        )

        # Extract QA metadata
        self.last_quality_score = int(data.get("quality_score", 85))
        self.last_review_issues = data.get("review_issues", [])
        self.last_uncertainty_warnings = data.get("uncertainty_warnings", [])

        # Append QA summary to recommendations for frontend visibility
        qa_summary = "\n\n[QUALITY ASSESSMENT]\nScore: {score}/100\nIssues: {issues}\nUncertainties: {warnings}".format(
            score=self.last_quality_score,
            issues=" | ".join(self.last_review_issues) if self.last_review_issues else "None",
            warnings=" | ".join(self.last_uncertainty_warnings) if self.last_uncertainty_warnings else "None"
        )
        draft.recommendations += qa_summary

        return self._apply_deterministic_qa(draft, state)

    def _apply_deterministic_qa(self, draft: ReportSection, state: PipelineState) -> ReportSection:
        """Hard-rule safety checks that cannot be overridden by LLM output."""
        # 1. Enforce exact disclaimer
        if draft.disclaimer != self.STANDARD_DISCLAIMER:
            draft.disclaimer = self.STANDARD_DISCLAIMER
            if self.last_review_issues:
                self.last_review_issues.append("Disclaimer corrected to regulatory standard.")
            else:
                self.last_review_issues = ["Disclaimer corrected to regulatory standard."]

        # 2. Cap quality score if critical pipeline failures occurred
        critical_failures = [
            k for k, v in state.agent_statuses.items() 
            if v == AgentStatus.ERROR and k in ["VISION", "REPORT"]
        ]
        if critical_failures:
            self.last_quality_score = min(self.last_quality_score, 40)
            self.last_review_issues.append(f"Pipeline degraded: {', '.join(critical_failures)} agents failed.")

        # 3. Flag low-confidence vision findings if not already warned
        if state.vision_output:
            low_conf_findings = [
                f.anatomical_region for f in state.vision_output.findings 
                if f.confidence.value == "LOW" or f.confidence_score < 50.0
            ]
            if low_conf_findings:
                warning = f"Low confidence observations in: {', '.join(low_conf_findings)}. Confirmatory imaging recommended."
                if warning not in self.last_uncertainty_warnings:
                    self.last_uncertainty_warnings.append(warning)

        # 4. Re-append updated QA summary if modified
        qa_summary = "\n\n[QUALITY ASSESSMENT]\nScore: {score}/100\nIssues: {issues}\nUncertainties: {warnings}".format(
            score=self.last_quality_score,
            issues=" | ".join(self.last_review_issues) if self.last_review_issues else "None",
            warnings=" | ".join(self.last_uncertainty_warnings) if self.last_uncertainty_warnings else "None"
        )
        if qa_summary not in draft.recommendations:
            draft.recommendations += qa_summary

        logger.info(f"✅ Critic Agent completed | QA Score: {self.last_quality_score}/100")
        return draft

    def _get_fallback_report(self, draft: ReportSection) -> ReportSection:
        """Safe fallback when critic review cannot be completed."""
        draft.disclaimer = self.STANDARD_DISCLAIMER
        self.last_quality_score = 50
        self.last_review_issues = ["Critic agent unavailable. Report delivered unreviewed."]
        self.last_uncertainty_warnings = ["Peer review skipped. Manual radiologist verification mandatory."]
        return self._apply_deterministic_qa(draft, PipelineState())
