from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT
from app.errors import ErrorKind, ReviewError
from app.models import ModelResponse, ModelSpec, Submission
from app.openrouter import MockOpenRouterClient
from app.pipeline import ReviewPipeline
from app.router import ModelRouter
from app.scenarios import load_scenarios
from app.storage import Storage


class FlakyFirstReviewerClient(MockOpenRouterClient):
    async def complete(self, spec: ModelSpec, messages, **kwargs) -> ModelResponse:
        if spec.model == "deepseek/deepseek-chat" and "scenario=copy_review" in str(messages):
            raise ReviewError(ErrorKind.RATE_LIMIT, "模拟限流", model=spec.model, retry_after=0)
        return await super().complete(spec, messages, **kwargs)


class InvalidOutputClient(MockOpenRouterClient):
    async def complete(self, spec, messages, **kwargs):
        return ModelResponse(model=spec.model, content="upstream error", prompt_tokens=10,
                             completion_tokens=2, cost_usd=0.001, latency_ms=1)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="debate-room-test-")
        self.storage = Storage(Path(self.temp.name) / "test.db")
        self.scenarios = load_scenarios(PROJECT_ROOT / "config" / "scenarios.yaml")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def pipeline(self, client=None) -> ReviewPipeline:
        return ReviewPipeline(
            self.storage,
            ModelRouter(self.storage),
            client or MockOpenRouterClient(),
            self.scenarios,
            max_retries_per_model=0,
        )

    async def run_task(self, scenario: str, content: str, client=None, **kwargs):
        task_id, _ = self.storage.create_task(scenario, Submission(content=content, **kwargs))
        await self.pipeline(client).process(task_id)
        return self.storage.get_task_detail(task_id)

    async def test_copy_review_complete(self) -> None:
        detail = await self.run_task("copy_review", "全新产品，欢迎体验。")
        self.assertEqual(detail["status"], "done")
        self.assertEqual(len(detail["model_outputs"]), 4)
        self.assertIn("score", detail["verdict"])
        self.assertEqual(detail["verdict"]["judge_model"], "anthropic/claude-3.5-sonnet")

    async def test_invalid_output_is_not_a_success(self) -> None:
        detail = await self.run_task("copy_review", "内容", client=InvalidOutputClient())
        self.assertEqual(detail["status"], "failed")
        self.assertIsNone(detail["verdict"])
        self.assertTrue(any(e["error_type"] == "invalid_output" for e in detail["errors"]))
        self.assertGreater(detail["total_cost_usd"], 0)

    async def test_code_review_complete(self) -> None:
        detail = await self.run_task("code_review", "diff --git a/a.py b/a.py\n+password = input()")
        self.assertEqual(detail["status"], "done")
        self.assertTrue(detail["verdict"]["issues"])

    async def test_fact_check_consensus(self) -> None:
        detail = await self.run_task("fact_check", "一个缺乏来源的说法")
        self.assertEqual(detail["status"], "done")
        self.assertTrue(detail["verdict"]["needs_human"])
        self.assertEqual(detail["verdict"]["confidence"], "low")
        self.assertEqual(detail["verdict"]["stance"], "insufficient_evidence")
        self.assertEqual(detail["verdict"]["evidence_status"], "disabled")

    async def test_fact_check_disagreement_goes_to_human(self) -> None:
        detail = await self.run_task("fact_check", "触发分歧演示：某个说法")
        self.assertEqual(detail["status"], "done")
        self.assertTrue(detail["verdict"]["needs_human"])
        self.assertIsNone(detail["verdict"]["answer"])

    async def test_opposite_stances_with_near_identical_answers_need_review(self) -> None:
        def review(answer, stance):
            return ModelResponse(model="test", content=answer, prompt_tokens=0,
                                 completion_tokens=0, cost_usd=0, latency_ms=0,
                                 parsed={"answer": answer, "stance": stance})
        reviews = [
            review("研究结果表明该药物能显著降低患者的死亡风险。", "supported"),
            review("研究结果表明该药物能显著降低患者的死亡风险。", "supported"),
            review("研究结果表明该药物不能显著降低患者的死亡风险。", "refuted"),
        ]
        verdict = self.pipeline()._cluster_fact_check(self.scenarios["fact_check"], reviews)
        self.assertTrue(verdict["needs_human"])
        self.assertIsNone(verdict["answer"])
        self.assertEqual(verdict["clusters"], [["R1", "R2"], ["R3"]])
        self.assertEqual(verdict["confidence"], "low")

    async def test_missing_or_invalid_stance_needs_review(self) -> None:
        for stance in (None, "unknown", ["supported"]):
            with self.subTest(stance=stance):
                reviews = [ModelResponse(model="test", content="相同回答", prompt_tokens=0,
                          completion_tokens=0, cost_usd=0, latency_ms=0,
                          parsed={"answer": "相同回答", "stance": value})
                           for value in ("supported", "supported", stance)]
                verdict = self.pipeline()._cluster_fact_check(self.scenarios["fact_check"], reviews)
                self.assertTrue(verdict["needs_human"])
                self.assertIsNone(verdict["answer"])

    async def test_rate_limit_falls_back(self) -> None:
        detail = await self.run_task(
            "copy_review", "测试降级链", client=FlakyFirstReviewerClient()
        )
        self.assertEqual(detail["status"], "done")
        self.assertTrue(any(error["error_type"] == "rate_limit" for error in detail["errors"]))
        self.assertTrue(
            any(output["model"] == "qwen/qwen-2.5-72b-instruct" for output in detail["model_outputs"])
        )

    async def test_tiny_budget_fails_explicitly(self) -> None:
        detail = await self.run_task(
            "copy_review", "预算测试", constraints={"max_cost_usd": 0.000001, "max_latency_ms": 10000}
        )
        self.assertEqual(detail["status"], "failed")
        self.assertIn("预算", detail["error"])


if __name__ == "__main__":
    unittest.main()
