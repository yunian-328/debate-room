from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT
from app.evidence import BraveEvidenceSearch, Source
from app.models import ModelResponse, Submission
from app.openrouter import MockOpenRouterClient
from app.pipeline import ReviewPipeline
from app.router import ModelRouter
from app.scenarios import load_scenarios
from app.storage import Storage


class SearchAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_fixed_endpoint_and_keeps_only_distinct_https_sources(self):
        def handler(request):
            self.assertEqual(request.url.host, "api.search.brave.com")
            self.assertEqual(request.url.params["search_lang"], "zh-hans")
            self.assertEqual(request.url.params["count"], "5")
            return httpx.Response(200, json={"web": {"results": [
                {"url": "https://example.org/a", "title": "<b>Title</b>", "description": "Good &amp; real"},
                {"url": "https://example.org/a#section", "description": "duplicate"},
                {"url": "javascript:alert(1)", "description": "invalid"},
                {"url": "https://example.org/b", "description": ""},
            ]}})
        search = BraveEvidenceSearch("testing", transport=httpx.MockTransport(handler))
        self.assertEqual(await search.search("示例", "zh", timeout=1), [
            Source("S1", "https://example.org/a", "Title", "Good & real")
        ])


class StaticSearch:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    async def search(self, claim, language, *, timeout):
        self.calls += 1
        if self.mode == "unavailable":
            raise httpx.ReadTimeout("simulated")
        if self.mode == "empty":
            return []
        return [Source("S1", "https://example.org/source", "Original source", "The reported value was 200 in 2025.")]


class ReviewerClient(MockOpenRouterClient):
    def __init__(self, ids):
        self.ids = ids
        self.inputs = []

    async def complete(self, spec, messages, **kwargs):
        import json
        self.inputs.append(messages[-1]["content"])
        return ModelResponse(
            spec.model, json.dumps({"answer": "The report says 200.", "stance": "supported", "evidence_ids": self.ids}),
            1, 1, 0, 1,
        )


class FactEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_fact(self, search, reviewer):
        with tempfile.TemporaryDirectory() as root:
            storage = Storage(Path(root) / "test.db")
            scenarios = load_scenarios(PROJECT_ROOT / "config" / "scenarios.yaml")
            pipeline = ReviewPipeline(storage, ModelRouter(storage), reviewer, scenarios, evidence_search=search)
            task_id, _ = storage.create_task("fact_check", Submission(content="The 2025 report says 200."))
            await pipeline.process(task_id)
            return storage.get_task_detail(task_id)["verdict"]

    async def test_shared_source_is_reused_and_traceable(self):
        search = StaticSearch("found")
        client = ReviewerClient(["S1"])
        verdict = await self.run_fact(search, client)
        self.assertEqual(search.calls, 1)
        self.assertEqual(len(client.inputs), 3)
        self.assertEqual(len(set(client.inputs)), 1)
        self.assertIn("reported value was 200", client.inputs[0])
        self.assertEqual(verdict["stance"], "supported")
        self.assertEqual(verdict["confidence"], "medium")
        self.assertEqual(verdict["cited_source_ids"], ["S1"])
        self.assertEqual(verdict["sources"][0]["url"], "https://example.org/source")

    async def test_invented_source_does_not_turn_into_fact(self):
        verdict = await self.run_fact(StaticSearch("found"), ReviewerClient(["S99"]))
        self.assertEqual(verdict["stance"], "insufficient_evidence")
        self.assertIsNone(verdict["answer"])
        self.assertTrue(verdict["needs_human"])

    async def test_empty_and_outage_are_not_false(self):
        for mode in ("empty", "unavailable"):
            with self.subTest(mode=mode):
                verdict = await self.run_fact(StaticSearch(mode), ReviewerClient(["S1"]))
                self.assertEqual(verdict["evidence_status"], mode)
                # 无来源进入记忆模式：评审立场一致即接受结论（置信度有限），
                # 而不是把缺证据误判为“证伪”。
                self.assertEqual(verdict["stance"], "supported")
                self.assertEqual(verdict["confidence"], "medium")
