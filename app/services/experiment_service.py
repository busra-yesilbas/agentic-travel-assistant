"""
Experiment / evaluation service.

Runs the full TripGenie pipeline against a set of sample queries and
computes summary evaluation metrics. Used for offline evaluation and
the /eval/run endpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.eval_metrics import EvaluationMetrics
from app.schemas.domain import TripIntent
from app.schemas.requests import TripPlanningRequest
from app.schemas.responses import EvalQueryResult, EvalRunResponse

logger = get_logger(__name__)


class ExperimentService:
    """
    Offline evaluation harness for the TripGenie assistant.

    Loads sample queries from data/sample_user_queries.json, runs the
    agent workflow for each, and computes aggregate quality metrics.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._metrics = EvaluationMetrics()

    def load_sample_queries(self, n: int = 10) -> list[dict]:
        """Load up to n sample queries from the dataset."""
        path = self._settings.sample_queries_path
        if not path.exists():
            logger.warning("eval.sample_queries_not_found", path=str(path))
            return self._synthetic_queries()

        with path.open() as fh:
            queries = json.load(fh)

        return queries[:n]

    async def run_evaluation(
        self, num_samples: int = 10, save_results: bool = False
    ) -> EvalRunResponse:
        """
        Run evaluation across sample queries and return summary metrics.

        This method imports the workflow lazily to avoid circular imports.
        """
        from app.agents.workflow import TripPlanningWorkflow

        workflow = TripPlanningWorkflow()
        sample_queries = self.load_sample_queries(n=num_samples)

        results: list[EvalQueryResult] = []
        successful = 0
        failed = 0

        for sample in sample_queries:
            query_id = sample.get("id", f"query_{len(results)+1:03d}")
            query_text = sample.get("query", "")
            expected_city = sample.get("expected_city", "")
            expected_days = sample.get("expected_days", 3)
            expected_budget = sample.get("expected_budget", "mid")
            key_interests = sample.get("key_interests", [])

            logger.info("eval.query_start", query_id=query_id)
            start = time.perf_counter()

            try:
                request = TripPlanningRequest(
                    query=query_text,
                    city=expected_city or None,
                    use_llm=True,
                )
                state = await workflow.run(request)

                elapsed_ms = (time.perf_counter() - start) * 1000

                # Compute per-query metrics
                constraint_sat = self._metrics.constraint_satisfaction(
                    intent=state.parsed_intent,
                    expected_city=expected_city,
                    expected_days=expected_days,
                    expected_budget=expected_budget,
                )
                rec_relevance = self._metrics.recommendation_relevance(
                    ranked_hotels=state.ranked_hotels,
                    intent=state.parsed_intent,
                    key_interests=key_interests,
                )
                completeness = self._metrics.itinerary_completeness(
                    itinerary=state.itinerary,
                    expected_days=expected_days,
                )
                helpfulness = self._metrics.answer_helpfulness(
                    final_answer=state.final_answer or "",
                    intent=state.parsed_intent,
                )
                overall = (constraint_sat + rec_relevance + completeness + helpfulness) / 4.0

                results.append(
                    EvalQueryResult(
                        query_id=query_id,
                        query=query_text[:100],
                        city=expected_city,
                        latency_ms=round(elapsed_ms, 1),
                        constraint_satisfaction=round(constraint_sat, 3),
                        recommendation_relevance=round(rec_relevance, 3),
                        itinerary_completeness=round(completeness, 3),
                        helpfulness_score=round(helpfulness, 3),
                        overall_score=round(overall, 3),
                    )
                )
                successful += 1
                logger.info(
                    "eval.query_done",
                    query_id=query_id,
                    overall=round(overall, 3),
                    latency_ms=round(elapsed_ms, 1),
                )

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error("eval.query_failed", query_id=query_id, error=str(exc))
                results.append(
                    EvalQueryResult(
                        query_id=query_id,
                        query=query_text[:100],
                        city=expected_city,
                        latency_ms=round(elapsed_ms, 1),
                        constraint_satisfaction=0.0,
                        recommendation_relevance=0.0,
                        itinerary_completeness=0.0,
                        helpfulness_score=0.0,
                        overall_score=0.0,
                        error=str(exc)[:200],
                    )
                )
                failed += 1

        # Aggregate metrics (successful queries only)
        successful_results = [r for r in results if r.error is None]
        n_ok = len(successful_results)

        def _avg(field: str) -> float:
            if not successful_results:
                return 0.0
            return sum(getattr(r, field) for r in successful_results) / n_ok

        response = EvalRunResponse(
            num_queries=len(results),
            successful=successful,
            failed=failed,
            avg_latency_ms=round(_avg("latency_ms"), 1),
            avg_constraint_satisfaction=round(_avg("constraint_satisfaction"), 3),
            avg_recommendation_relevance=round(_avg("recommendation_relevance"), 3),
            avg_itinerary_completeness=round(_avg("itinerary_completeness"), 3),
            avg_helpfulness_score=round(_avg("helpfulness_score"), 3),
            avg_overall_score=round(_avg("overall_score"), 3),
            results=results,
            summary=self._build_summary(successful, failed, _avg("overall_score")),
        )

        if save_results:
            self._save_results(response)

        return response

    def _build_summary(self, successful: int, failed: int, avg_score: float) -> str:
        label = "Excellent" if avg_score >= 0.85 else "Good" if avg_score >= 0.70 else "Fair"
        return (
            f"Evaluation complete: {successful} queries succeeded, {failed} failed. "
            f"Average overall score: {avg_score:.3f} ({label})."
        )

    def _save_results(self, response: EvalRunResponse) -> None:
        output_path = Path(self._settings.project_root) / "data" / "eval_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(response.model_dump(), fh, indent=2)
        logger.info("eval.results_saved", path=str(output_path))

    @staticmethod
    def _synthetic_queries() -> list[dict]:
        """Minimal fallback if sample query file is missing."""
        return [
            {
                "id": "synthetic_001",
                "query": "4-day Amsterdam trip for a couple, mid-range budget, museums and canals.",
                "expected_city": "Amsterdam",
                "expected_days": 4,
                "expected_budget": "mid",
                "key_interests": ["museums", "canals"],
            },
            {
                "id": "synthetic_002",
                "query": "3-day Paris trip, romantic, luxury budget.",
                "expected_city": "Paris",
                "expected_days": 3,
                "expected_budget": "luxury",
                "key_interests": ["romance", "art"],
            },
        ]
