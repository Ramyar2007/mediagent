# mediagent/core/pipeline.py
"""
Pipeline Orchestrator for MediAgent.
Manages sequential execution of all medical imaging analysis agents,
tracks real-time state, handles graceful error recovery, and coordinates
data flow between specialized AI components.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from core.models import (
    AgentStatus,
    FinalReport,
    IntakeOutput,
    PipelineState,
    PatientInput,
    ResearchOutput,
    ReportSection,
    VisionOutput,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Central execution engine that routes patient data through the 5-agent
    medical analysis pipeline. Maintains real-time state, supports streaming
    status callbacks for UI synchronization, and implements graceful degradation
    on partial agent failures.
    """

    AGENT_ORDER = ["INTAKE", "VISION", "RESEARCH", "REPORT", "CRITIC"]

    def __init__(
        self,
        intake_agent: Any,
        vision_agent: Any,
        research_agent: Any,
        report_agent: Any,
        critic_agent: Any,
        on_status_update: Optional[Callable[[PipelineState], None]] = None,
    ):
        """
        Initialize orchestrator with instantiated agents and optional status callback.
        
        Args:
            intake_agent: Instance of IntakeAgent
            vision_agent: Instance of VisionAgent
            research_agent: Instance of ResearchAgent
            report_agent: Instance of ReportAgent
            critic_agent: Instance of CriticAgent
            on_status_update: Optional callback invoked after each agent completes
        """
        self.agents = {
            "INTAKE": intake_agent,
            "VISION": vision_agent,
            "RESEARCH": research_agent,
            "REPORT": report_agent,
            "CRITIC": critic_agent,
        }
        self.on_status_update = on_status_update

    def run(self, patient_input: PatientInput) -> PipelineState:
        """
        Execute the full diagnostic pipeline sequentially.
        
        Args:
            patient_input: Validated patient submission with image & context
            
        Returns:
            PipelineState: Complete execution state containing all outputs,
                          agent statuses, and final report.
        """
        logger.info("🚀 Pipeline execution started | Input ID: %s", id(patient_input))
        state = PipelineState()
        
        try:
            # ─────────────────────────────────────────────────────────────
            # STEP 1: INTAKE AGENT
            # ─────────────────────────────────────────────────────────────
            self._execute_step(state, "INTAKE", patient_input=patient_input)
            if state.agent_statuses["INTAKE"] == AgentStatus.ERROR:
                logger.warning("⚠️ Intake failed. Halting pipeline for safety.")
                return state

            # ─────────────────────────────────────────────────────────────
            # STEP 2: VISION AGENT
            # ─────────────────────────────────────────────────────────────
            self._execute_step(
                state, "VISION",
                image_base64=patient_input.image_base64,
                clinical_context=state.intake_output.standardized_symptoms
            )
            if state.agent_statuses["VISION"] == AgentStatus.ERROR:
                logger.warning("⚠️ Vision analysis failed. Continuing with degraded pipeline.")

            # ─────────────────────────────────────────────────────────────
            # STEP 3: RESEARCH AGENT
            # ─────────────────────────────────────────────────────────────
            self._execute_step(
                state, "RESEARCH",
                vision_findings=state.vision_output.findings if state.vision_output else [],
                demographics=state.intake_output.extracted_demographics
            )
            if state.agent_statuses["RESEARCH"] == AgentStatus.ERROR:
                logger.warning("⚠️ Research failed. Generating report without differential augmentation.")

            # ─────────────────────────────────────────────────────────────
            # STEP 4: REPORT AGENT
            # ─────────────────────────────────────────────────────────────
            self._execute_step(
                state, "REPORT",
                intake=state.intake_output,
                vision=state.vision_output,
                research=state.research_output
            )
            if state.agent_statuses["REPORT"] == AgentStatus.ERROR:
                logger.error("❌ Report generation failed. Pipeline cannot complete safely.")
                return state

            # ─────────────────────────────────────────────────────────────
            # STEP 5: CRITIC AGENT
            # ─────────────────────────────────────────────────────────────
            self._execute_step(
                state, "CRITIC",
                draft_report=state.report_draft,
                pipeline_state=state
            )
            
            # Assemble final report
            state.final_report = self._assemble_final_report(state)
            state.current_step = "COMPLETE"
            
            logger.info("✅ Pipeline execution completed successfully.")

        except Exception as e:
            logger.exception("💥 Unhandled pipeline failure: %s", str(e))
            state.current_step = "FAILED"
            state.error_log.append(f"SYSTEM_FAILURE: {str(e)}")

        return state

    def _execute_step(
        self,
        state: PipelineState,
        agent_name: str,
        **kwargs
    ) -> None:
        """
        Generic step executor with state management, timing, and error isolation.
        """
        logger.info(f"▶️ Executing agent: {agent_name}")
        state.current_step = agent_name
        state.mark_running(agent_name)
        self._notify(state)

        start_time = time.perf_counter()
        try:
            agent = self.agents[agent_name]
            output = agent.process(**kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Store output in state
            if output is not None:
                if agent_name == "INTAKE":
                    state.intake_output = output
                elif agent_name == "VISION":
                    state.vision_output = output
                elif agent_name == "RESEARCH":
                    state.research_output = output
                elif agent_name == "REPORT":
                    state.report_draft = output
                    
            state.mark_done(agent_name)
            logger.info(f"✅ {agent_name} completed in {elapsed:.3f}s")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_msg = f"{agent_name} execution failed after {elapsed:.3f}s: {str(e)}"
            logger.error("❌ %s", error_msg, exc_info=True)
            state.mark_error(agent_name, str(e))

        finally:
            self._notify(state)

    def _assemble_final_report(self, state: PipelineState) -> FinalReport:
        """
        Synthesize all agent outputs into the final deliverable report structure.
        Applies critic modifications and standardizes formatting.
        """
        report_draft = state.report_draft or ReportSection()
        
        # Determine overall severity from vision findings
        overall_severity = "NORMAL"
        if state.vision_output and state.vision_output.findings:
            severity_hierarchy = {"CRITICAL": 4, "SIGNIFICANT": 3, "INCIDENTAL": 2, "NORMAL": 1}
            highest = max(
                (severity_hierarchy.get(f.severity.value, 1) for f in state.vision_output.findings),
                default=1
            )
            severity_map = {4: "CRITICAL", 3: "SIGNIFICANT", 2: "INCIDENTAL", 1: "NORMAL"}
            overall_severity = severity_map.get(highest, "NORMAL")

        # Build vision summary
        vision_summary = "No imaging analysis performed."
        if state.vision_output:
            anomalies = [f.description for f in state.vision_output.findings if f.is_anomaly]
            vision_summary = (
                f"Modality: {state.vision_output.modality_detected.value} | "
                f"Quality: {state.vision_output.technical_quality} | "
                f"Anomalies Detected: {len(anomalies)} | "
                f"Overall: {state.vision_output.overall_assessment}"
            )

        # Build research summary
        research_summary = "No differential diagnosis generated."
        if state.research_output and state.research_output.differential_diagnoses:
            top_dx = state.research_output.differential_diagnoses[:3]
            dx_list = " | ".join([d.condition_name for d in top_dx])
            research_summary = f"Top Differentials: {dx_list} | Confidence: {'/'.join([f'{d.match_probability:.0f}%' for d in top_dx])}"

        return FinalReport(
            patient_metadata=state.intake_output.extracted_demographics if state.intake_output else {},
            sections=report_draft,
            vision_summary=vision_summary,
            research_summary=research_summary,
            overall_severity=overall_severity,
            agent_pipeline_status=state.agent_statuses,
            generation_timestamp=datetime.now()
        )

    def _notify(self, state: PipelineState) -> None:
        """Invoke status callback if registered (used for SSE/UI polling)."""
        if self.on_status_update:
            try:
                self.on_status_update(state)
            except Exception as e:
                logger.warning("⚠️ Status callback failed: %s", str(e))
