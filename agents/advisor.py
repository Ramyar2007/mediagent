# mediagent/agents/advisor.py
"""
Clinical Advisor Agent for MediAgent.
Post-report interactive Q&A. Answers follow-up clinical questions
from radiologists/clinicians in full context of the generated report.
Acts as a 24/7 senior radiology consultant.
"""

import logging
import re
from typing import Optional

from core.llm import LLMClient
from core.models import FinalReport

logger = logging.getLogger(__name__)


class ClinicalAdvisorAgent:
    """
    Interactive clinical consultation agent activated after report generation.
    Answers follow-up questions with access to all pipeline outputs.
    Scope is limited to radiological interpretation — no treatment prescriptions.
    """

    SYSTEM_PROMPT = """You are a senior radiologist consultant. Answer the clinician's question directly and concisely — 2-4 sentences maximum. No preamble, no thinking out loud, no reasoning steps. Just the answer.
Rules: reference report findings; no fabrication; no medications/dosages; formal radiological tone. If management decision, end with "Clinical correlation recommended." """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def answer(self, question: str, report: FinalReport) -> str:
        """
        Answer a follow-up clinical question in the context of the generated report.

        Args:
            question: Free-text clinical question from the user
            report: The FinalReport from the pipeline

        Returns:
            str: Clinical answer text
        """
        logger.info("💬 Clinical Advisor processing question: %.80s", question)

        sections = report.sections
        severity = report.overall_severity.value if hasattr(report.overall_severity, "value") else str(report.overall_severity)

        # Send only the most relevant report fields — less tokens = faster response
        context = (
            f"Severity: {severity} | Impression: {sections.impression} | "
            f"Findings: {sections.findings[:600]} | "
            f"Recommendations: {sections.recommendations[:300]}"
        )

        prompt = f"Report: {context}\n\nQuestion: {question}\n\nAnswer:"

        result = self.llm.generate_text(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
            # Disable Qwen3 thinking/reasoning mode entirely
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        if result.get("success") and result.get("content"):
            answer = self._strip_thinking(result["content"].strip())
            logger.info("✅ Clinical Advisor answered | tokens=%s", result.get("usage", {}).get("total_tokens", 0))
            return answer

        logger.warning("⚠️ Clinical Advisor LLM call failed")
        return "Unable to process this question. Please review the report directly and consult a licensed radiologist."

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Remove all thinking/reasoning output that Qwen and similar models emit.
        Handles both structured tags and plain-text reasoning patterns.
        """
        # Remove <think>...</think> XML blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove ```think ... ``` markdown blocks
        text = re.sub(r"```think.*?```", "", text, flags=re.DOTALL)
        # Remove plain-text reasoning preambles Qwen3 emits without tags:
        # "Here's a thinking process: 1. ..." or "Let me think: ..." etc.
        text = re.sub(
            r"(?i)^(here'?s?\s+(a\s+)?thinking\s+process:?|let me (think|analyze|consider):?|thinking:?).*?(\n\n|\Z)",
            "", text, flags=re.DOTALL
        )
        # Remove numbered reasoning lists at the start (1. **Title:** ...)
        text = re.sub(r"^(\s*\d+\.\s+\*\*[^*]+\*\*:.*\n?)+", "", text, flags=re.MULTILINE)
        # If after stripping we have a clear section break, take only what's after it
        if "\n\n" in text:
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            # Take the last substantial chunk (the actual answer)
            for part in reversed(parts):
                if len(part) > 20:
                    return part
        return text.strip()
