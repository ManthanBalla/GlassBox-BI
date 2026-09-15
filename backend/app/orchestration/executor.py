"""Sequential Workflow Executor for Multi-Agent Orchestration (Phase 8).

Executes the pipeline stages sequentially, safely passes typed data contracts between stages,
enforces error isolation boundaries, and guarantees zero automatic retries or self-corrections.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping, auto_detect_column_mapping
from backend.app.schemas.decisions import DecisionRequest, DecisionResult
from backend.app.schemas.evaluation import BenchmarkResult, EvaluationRequest
from backend.app.schemas.explainability import ExplanationResult, LocalExplanationRequest
from backend.app.schemas.forecasting import ForecastRequest, ForecastResult
from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    PipelineStage,
    StageExecutionResult,
    StageStatus,
    WorkflowStatus,
)
from backend.app.schemas.processing import DataProcessingConfig, DataProcessingResult
from backend.app.orchestration.audit import WorkflowAuditTracker
from backend.app.orchestration.graph import WorkflowGraph
from backend.app.orchestration.registry import AgentRegistry


class SequentialWorkflowExecutor:
    """Coordinates deterministic, stage-by-stage multi-agent execution."""

    def __init__(self, registry: Optional[AgentRegistry] = None) -> None:
        self.registry = registry or AgentRegistry()

    def _resolve_default_dataset_path(self, requested_path: Optional[str]) -> str:
        """Resolves an accessible benchmark dataset path if none was explicitly provided."""
        if requested_path and Path(requested_path).exists():
            return requested_path

        candidates = [
            "data/processed/synthetic/retail_processed.csv",
            "data/sample/retail_sample.csv",
            "data/sample/retail_processed_sample.csv",
        ]
        for c in candidates:
            if Path(c).exists():
                return c

        raise FileNotFoundError(
            f"No valid dataset found. Requested: '{requested_path}'. "
            f"Checked fallbacks: {candidates}"
        )

    def execute(
        self,
        request: OrchestrationRequest,
        state: OrchestrationState,
    ) -> OrchestrationResult:
        """Executes the complete multi-agent workflow sequentially."""
        start_time_perf = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        state.status = WorkflowStatus.RUNNING
        state.timestamps["started_at"] = started_at.isoformat()

        # Shared execution context across stages
        context: Dict[str, Any] = {
            "dataset_path": None,
            "raw_df": None,
            "processed_df": None,
            "data_result": None,
            "forecast_result": None,
            "evaluation_result": None,
            "explanation_result": None,
            "decision_result": None,
        }

        # Canonical sequence
        stages = WorkflowGraph.get_sequence()

        for idx, stage in enumerate(stages):
            state.current_stage = stage

            # Check eligibility against dependency graph
            can_run, reason = WorkflowGraph.can_execute(
                stage=stage,
                completed_stages=state.completed_stages,
                failed_stages=state.failed_stages,
            )

            if not can_run:
                # Mark as skipped
                skipped_res = StageExecutionResult(
                    stage=stage,
                    status=StageStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=0.0,
                    warnings=[f"Stage '{stage.value}' skipped: {reason}"],
                )
                state.stage_results[stage.value] = skipped_res
                state.skipped_stages.append(stage)
                state.warnings.append(f"Stage '{stage.value}' was skipped: {reason}")
                continue

            # Execute the stage
            stage_start_perf = time.perf_counter()
            stage_started_at = datetime.now(timezone.utc)

            try:
                if stage == PipelineStage.DATA_PROCESSING:
                    self._execute_data_processing(request, context, state)
                elif stage == PipelineStage.FORECASTING:
                    self._execute_forecasting(request, context, state)
                elif stage == PipelineStage.EVALUATION:
                    self._execute_evaluation(request, context, state)
                elif stage == PipelineStage.EXPLAINABILITY:
                    self._execute_explainability(request, context, state)
                elif stage == PipelineStage.DECISION_INTELLIGENCE:
                    self._execute_decision_intelligence(request, context, state)

                # If execution completed without raising, record SUCCESS
                duration_ms = (time.perf_counter() - stage_start_perf) * 1000
                stage_record = state.stage_results.get(stage.value)
                if stage_record:
                    stage_record.completed_at = datetime.now(timezone.utc)
                    stage_record.duration_ms = round(duration_ms, 2)
                    stage_record.status = StageStatus.SUCCESS
                state.completed_stages.append(stage)

            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_start_perf) * 1000
                err_msg = f"Stage '{stage.value}' failed: {str(exc)}"
                state.errors.append(err_msg)
                state.failed_stages.append(stage)

                failed_rec = StageExecutionResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    started_at=stage_started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=round(duration_ms, 2),
                    errors=[err_msg],
                )
                state.stage_results[stage.value] = failed_rec

                # If blocking failure, skip all remaining downstream stages
                if WorkflowGraph.is_blocking_failure(stage):
                    remaining = stages[idx + 1 :]
                    to_skip = WorkflowGraph.get_stages_to_skip_after_failure(stage, remaining)
                    for skip_stage in to_skip:
                        skip_rec = StageExecutionResult(
                            stage=skip_stage,
                            status=StageStatus.SKIPPED,
                            started_at=datetime.now(timezone.utc),
                            completed_at=datetime.now(timezone.utc),
                            duration_ms=0.0,
                            warnings=[f"Stage '{skip_stage.value}' skipped due to upstream blocking failure in '{stage.value}'."],
                        )
                        state.stage_results[skip_stage.value] = skip_rec
                        state.skipped_stages.append(skip_stage)
                        state.warnings.append(f"Stage '{skip_stage.value}' was skipped due to blocking failure in '{stage.value}'.")
                    break

        # Finalize status and audit
        completed_at = datetime.now(timezone.utc)
        total_duration_ms = (time.perf_counter() - start_time_perf) * 1000
        state.timestamps["completed_at"] = completed_at.isoformat()
        state.current_stage = None

        # Determine overall workflow status
        has_blocking_failure = any(WorkflowGraph.is_blocking_failure(s) for s in state.failed_stages)
        if has_blocking_failure or (state.failed_stages and not state.completed_stages):
            state.status = WorkflowStatus.FAILED
        elif state.failed_stages:
            state.status = WorkflowStatus.PARTIAL
        else:
            state.status = WorkflowStatus.COMPLETED

        # Build immutable audit record
        audit_record = WorkflowAuditTracker.build_audit_record(
            state=state,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration_ms,
        )

        # Stage summary dictionary
        stage_summary = {
            s.value: {
                "status": state.stage_results[s.value].status.value if s.value in state.stage_results else "NOT_RUN",
                "duration_ms": state.stage_results[s.value].duration_ms if s.value in state.stage_results else 0.0,
                "warnings": state.stage_results[s.value].warnings if s.value in state.stage_results else [],
                "errors": state.stage_results[s.value].errors if s.value in state.stage_results else [],
            }
            for s in stages
        }

        # High-level execution summary
        fcst_out: Optional[ForecastResult] = context.get("forecast_result")
        dec_out: Optional[DecisionResult] = context.get("decision_result")
        exec_summary = {
            "workflow_id": state.workflow_id,
            "status": state.status.value,
            "total_duration_ms": round(total_duration_ms, 2),
            "completed_stages": [s.value for s in state.completed_stages],
            "failed_stages": [s.value for s in state.failed_stages],
            "skipped_stages": [s.value for s in state.skipped_stages],
            "selected_model": fcst_out.model_name if fcst_out else None,
            "primary_recommendation": dec_out.primary_recommendation.action if dec_out and dec_out.primary_recommendation else None,
            "decision_confidence": dec_out.audit_trail.decision_confidence if dec_out and dec_out.audit_trail else None,
            "requires_human_review": dec_out.requires_human_review if dec_out else False,
        }
        state.execution_summary = exec_summary

        final_result = OrchestrationResult(
            workflow_id=state.workflow_id,
            status=state.status,
            stage_summary=stage_summary,
            data_processing=context.get("data_result").model_dump(mode="json") if context.get("data_result") else None,
            forecast=fcst_out,
            evaluation=context.get("evaluation_result").model_dump(mode="json") if context.get("evaluation_result") else None,
            explanation=context.get("explanation_result"),
            decisions=dec_out,
            warnings=list(state.warnings),
            errors=list(state.errors),
            audit=audit_record,
            execution_summary=exec_summary,
        )

        state.final_result = final_result
        return final_result

    # -------------------------------------------------------------------------
    # Stage 1: Data Processing
    # -------------------------------------------------------------------------
    def _execute_data_processing(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        state: OrchestrationState,
    ) -> None:
        resolved_path = self._resolve_default_dataset_path(request.dataset_path)
        context["dataset_path"] = resolved_path

        df = pd.read_csv(resolved_path)
        context["raw_df"] = df

        processor = self.registry.get_agent(PipelineStage.DATA_PROCESSING)
        processed_df, proc_result = processor.process(
            df=df,
            dataset_id=f"wf_{state.workflow_id}",
            dataset_name=Path(resolved_path).stem,
        )

        context["processed_df"] = processed_df
        context["data_result"] = proc_result

        meta = {
            "dataset_path": resolved_path,
            "input_rows": proc_result.input_row_count,
            "processed_rows": proc_result.output_row_count,
            "features_generated": len(proc_result.generated_features),
            "quality_score_before": proc_result.quality_score_before.overall_score if proc_result.quality_score_before else None,
            "quality_score_after": proc_result.quality_score_after.overall_score if proc_result.quality_score_after else None,
            "frequency": proc_result.time_series_integrity.expected_frequency if proc_result.time_series_integrity else "D",
        }

        stage_rec = StageExecutionResult(
            stage=PipelineStage.DATA_PROCESSING,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            output=proc_result,
            warnings=proc_result.warnings,
            metadata=meta,
        )
        state.stage_results[PipelineStage.DATA_PROCESSING.value] = stage_rec
        state.warnings.extend(proc_result.warnings)

    # -------------------------------------------------------------------------
    # Stage 2: Forecasting
    # -------------------------------------------------------------------------
    def _execute_forecasting(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        state: OrchestrationState,
    ) -> None:
        processed_df = context.get("processed_df")
        if processed_df is None:
            raise RuntimeError("Cannot run forecasting: processed_df is missing from context.")

        forecaster = self.registry.get_agent(PipelineStage.FORECASTING)
        fcst_req = ForecastRequest(
            dataset_id=f"wf_{state.workflow_id}",
            entity_id=request.entity_id,
            product_id=request.product_id,
            horizon=request.horizon,
            selection_metric=request.selection_metric,
            confidence_level=request.confidence_level,
            candidate_models=request.candidate_models,
        )

        fcst_result = forecaster.run_forecast(request=fcst_req, df=processed_df)
        context["forecast_result"] = fcst_result

        meta = {
            "selected_model": fcst_result.model_name,
            "horizon": fcst_result.horizon,
            "predictions_count": len(fcst_result.predictions),
            "validation_metrics": fcst_result.metrics,
            "candidate_ranking": fcst_result.model_selection.ranking if fcst_result.model_selection else [],
            "uncertainty_method": fcst_result.uncertainty_method,
        }

        stage_rec = StageExecutionResult(
            stage=PipelineStage.FORECASTING,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            output=fcst_result,
            warnings=fcst_result.warnings,
            metadata=meta,
        )
        state.stage_results[PipelineStage.FORECASTING.value] = stage_rec
        state.warnings.extend(fcst_result.warnings)

    # -------------------------------------------------------------------------
    # Stage 3: Evaluation
    # -------------------------------------------------------------------------
    def _execute_evaluation(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        state: OrchestrationState,
    ) -> None:
        if not request.run_evaluation:
            # Explicitly skipped by user configuration
            skip_rec = StageExecutionResult(
                stage=PipelineStage.EVALUATION,
                status=StageStatus.SKIPPED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_ms=0.0,
                warnings=["Evaluation skipped: run_evaluation=False."],
            )
            state.stage_results[PipelineStage.EVALUATION.value] = skip_rec
            state.skipped_stages.append(PipelineStage.EVALUATION)
            return

        processed_df = context.get("processed_df")
        if processed_df is None:
            raise RuntimeError("Cannot evaluate: processed_df is missing from context.")

        eval_horizon = request.evaluation_horizon or request.horizon
        benchmark_engine = self.registry.get_agent(PipelineStage.EVALUATION)
        eval_req = EvaluationRequest(
            dataset_path=context.get("dataset_path"),
            entity_id=request.entity_id,
            product_id=request.product_id,
            horizon=eval_horizon,
            candidate_models=request.candidate_models,
            confidence_level=request.confidence_level,
            random_seed=request.random_seed,
        )

        benchmark_result = benchmark_engine.run_benchmark(
            request=eval_req,
            df=processed_df,
        )
        context["evaluation_result"] = benchmark_result

        # Find metrics for Phase 4 winner
        selected_name = benchmark_result.phase4_selected_model
        winner_eval = next((m for m in benchmark_result.models if m.model_name == selected_name), None)

        meta = {
            "evaluation_horizon": benchmark_result.horizon,
            "phase4_selected_model": selected_name,
            "test_start_date": benchmark_result.test_period.get("start"),
            "test_end_date": benchmark_result.test_period.get("end"),
            "winner_test_mae": winner_eval.mae if winner_eval else None,
            "winner_test_rmse": winner_eval.rmse if winner_eval else None,
            "winner_test_mape": winner_eval.mape if winner_eval else None,
            "candidate_models_evaluated": [m.model_name for m in benchmark_result.models],
        }

        stage_rec = StageExecutionResult(
            stage=PipelineStage.EVALUATION,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            output=benchmark_result,
            warnings=[],
            metadata=meta,
        )
        state.stage_results[PipelineStage.EVALUATION.value] = stage_rec

    # -------------------------------------------------------------------------
    # Stage 4: Explainability
    # -------------------------------------------------------------------------
    def _execute_explainability(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        state: OrchestrationState,
    ) -> None:
        if not request.run_explainability:
            # Explicitly skipped by user configuration
            skip_rec = StageExecutionResult(
                stage=PipelineStage.EXPLAINABILITY,
                status=StageStatus.SKIPPED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_ms=0.0,
                warnings=["Explainability skipped: run_explainability=False."],
            )
            state.stage_results[PipelineStage.EXPLAINABILITY.value] = skip_rec
            state.skipped_stages.append(PipelineStage.EXPLAINABILITY)
            return

        fcst_result: Optional[ForecastResult] = context.get("forecast_result")
        if fcst_result is None:
            raise RuntimeError("Cannot run explainability: forecast_result is missing from context.")

        processed_df = context.get("processed_df")
        model_name = fcst_result.model_name
        explainer = self.registry.get_agent(PipelineStage.EXPLAINABILITY)

        local_req = LocalExplanationRequest(
            dataset_path=context.get("dataset_path"),
            model_name=model_name,
            entity_id=request.entity_id,
            product_id=request.product_id,
            method=request.explanation_method,
            prediction_index=0,
            random_seed=request.random_seed,
        )

        expl_result = explainer.explain_local(
            request=local_req,
            history_df=None,  # explain_local will partition leakage-safely from dataset
        )
        context["explanation_result"] = expl_result

        meta = {
            "model_name": model_name,
            "method": expl_result.method,
            "fidelity_score": expl_result.fidelity.fidelity_score if expl_result.fidelity else None,
            "explanation_status": expl_result.fidelity.explanation_status if expl_result.fidelity else None,
            "reconstruction_error": expl_result.fidelity.reconstruction_error if expl_result.fidelity else None,
            "top_positive_features": [c.feature for c in expl_result.top_positive_contributors[:3]] if expl_result.top_positive_contributors else [],
            "top_negative_features": [c.feature for c in expl_result.top_negative_contributors[:3]] if expl_result.top_negative_contributors else [],
        }

        stage_rec = StageExecutionResult(
            stage=PipelineStage.EXPLAINABILITY,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            output=expl_result,
            warnings=expl_result.warnings,
            metadata=meta,
        )
        state.stage_results[PipelineStage.EXPLAINABILITY.value] = stage_rec
        state.warnings.extend(expl_result.warnings)

    # -------------------------------------------------------------------------
    # Stage 5: Decision Intelligence
    # -------------------------------------------------------------------------
    def _execute_decision_intelligence(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        state: OrchestrationState,
    ) -> None:
        fcst_result: Optional[ForecastResult] = context.get("forecast_result")
        if fcst_result is None:
            raise RuntimeError("Cannot run decision intelligence: forecast_result is missing from context.")

        expl_result: Optional[ExplanationResult] = context.get("explanation_result")
        decision_agent = self.registry.get_agent(PipelineStage.DECISION_INTELLIGENCE)

        dec_req = DecisionRequest(
            forecast_result=fcst_result,
            explanation_result=expl_result,
            business_context=request.business_context,
            uncertainty_threshold=request.uncertainty_threshold,
            fidelity_threshold=request.fidelity_threshold,
            entity_id=request.entity_id,
            product_id=request.product_id,
        )

        dec_result = decision_agent.run_decision(request=dec_req)

        # Section 15 Policy: If Explainability failed or was omitted, explicitly record missing explanation
        if expl_result is None:
            missing_expl_msg = "Explainability stage was skipped or failed; decisions operating without feature attributions."
            if missing_expl_msg not in dec_result.warnings:
                dec_result.warnings.append(missing_expl_msg)
            if dec_result.primary_recommendation:
                if not any("explanation" in a.lower() for a in dec_result.primary_recommendation.assumptions):
                    dec_result.primary_recommendation.assumptions.append(
                        "No Phase 6 explanation provided; decisions based purely on forecast dynamics and business context."
                    )
                dec_result.primary_recommendation.requires_human_review = True
            dec_result.requires_human_review = True

        context["decision_result"] = dec_result

        meta = {
            "primary_action": dec_result.primary_recommendation.action if dec_result.primary_recommendation else None,
            "primary_priority": dec_result.primary_recommendation.priority.value if dec_result.primary_recommendation else None,
            "recommendations_count": len(dec_result.recommendations),
            "decision_confidence": dec_result.audit_trail.decision_confidence if dec_result.audit_trail else None,
            "requires_human_review": dec_result.requires_human_review,
            "rules_triggered": dec_result.rules_triggered,
            "explanation_available": expl_result is not None,
        }

        stage_rec = StageExecutionResult(
            stage=PipelineStage.DECISION_INTELLIGENCE,
            status=StageStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            output=dec_result,
            warnings=dec_result.warnings,
            metadata=meta,
        )
        state.stage_results[PipelineStage.DECISION_INTELLIGENCE.value] = stage_rec
        state.warnings.extend(dec_result.warnings)
