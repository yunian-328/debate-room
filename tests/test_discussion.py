from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.evidence import WikimediaEvidenceSearch
from app.main import create_app
from app.models import ModelResponse


class FakePanel:
    def __init__(self, *, synthetically_cited=True, partial=False, votes=(), stance="supported", invalid_slot=None, cited_id="S1"):
        self.calls = []
        self.synthetically_cited = synthetically_cited
        self.partial = partial
        self.votes = votes
        self.stance = stance
        self.invalid_slot = invalid_slot
        self.cited_id = cited_id

    async def complete(self, spec, messages, **kwargs):
        self.calls.append(messages)
        if (self.partial and len(self.calls) > 1) or len(self.calls) == self.invalid_slot:
            return ModelResponse(spec.model, "not JSON", 1, 1, 0.0, 1)
        # Deliberately overconfident in both branches. The aggregation layer still
        # keeps a unanimous opinion distinct from a traceable finding.
        encoded = json.dumps({
            "answer": "A mock panel opinion which does not establish the factual conclusion.",
            "stance": self.stance,
            "internal_vote": self.votes[len(self.calls)-1] if len(self.calls) <= len(self.votes) else None,
            "evidence_ids": [self.cited_id] if self.synthetically_cited else [],
        })
        return ModelResponse(spec.model, encoded, 1, 1, 0.0, 1)

    async def close(self):
        pass


class DiscussionApiTests(unittest.TestCase):
    def get_fact(self, search_response, panel, *, provider="wikimedia"):
        with tempfile.TemporaryDirectory() as temp:
            queries = []

            def wiki(request):
                queries.append(request)
                return search_response

            provider_fixture = WikimediaEvidenceSearch(transport=httpx.MockTransport(wiki))
            token = "private-preview-test-token-only-123"
            headers = {"Authorization": f"Bearer {token}"}
            settings = Settings(database_path=Path(temp) / "test.db", preview_api_token=token,
                                mock_mode=False, fact_search_provider=provider)
            with patch("app.main.WikimediaEvidenceSearch", return_value=provider_fixture), patch(
                "app.main.OpenRouterClient", return_value=panel
            ), TestClient(create_app(settings)) as client:
                form = client.post("/api/v1/hooks/fact-check", headers=headers,
                                   json={"content": "The 2025 example report shows 200 registrations.", "language": "en"})
                self.assertEqual(form.status_code, 202)
                for _ in range(100):
                    body = client.get(form.json()["detail_url"], headers=headers).json()
                    if body["status"] in {"done", "failed"}:
                        break
                    time.sleep(.01)
                self.assertEqual(body["status"], "done")
                intro = client.get("/workbench", headers=headers).text
                self.assertIn("内部讨论 · 未经证实", intro)
            return body["verdict"], queries

    def assert_unverified(self, verdict, discussed=3):
        self.assertEqual(verdict["stance"], "insufficient_evidence")
        self.assertIsNone(verdict["answer"])
        self.assertEqual(verdict["cited_source_ids"], [])
        self.assertTrue(verdict["needs_human"])
        self.assertEqual(len(verdict["discussion"]), discussed)
        self.assertIn("mock panel opinion", verdict["discussion"][0]["opinion"])

    def test_empty_and_unavailable_show_real_panel_output_but_fail_closed(self):
        for response, status in (
            (httpx.Response(200, json={"pages": []}), "empty"),
            (httpx.Response(503), "unavailable"),
        ):
            with self.subTest(status=status):
                panel = FakePanel()
                verdict, queries = self.get_fact(response, panel)
                self.assertEqual(len(queries), 1)
                self.assertEqual(len(panel.calls), 3)
                self.assertEqual(len({call[0]["content"] for call in panel.calls}), 3)
                self.assertEqual(verdict["evidence_status"], status)
                # 记忆模式（无来源）：全体立场一致即形成结论，置信度有限。
                self.assertEqual(verdict["stance"], "supported")
                self.assertEqual(verdict["confidence"], "medium")

    def test_unrelated_uncited_results_surface_opinion_not_evidence(self):
        fixture = httpx.Response(200, json={"pages": [
            {"id": 1, "key": "Water", "title": "Water", "excerpt": "Water is a chemical substance."}
        ]})
        panel = FakePanel(synthetically_cited=False)
        verdict, queries = self.get_fact(fixture, panel)
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(panel.calls), 3)
        self.assertEqual(verdict["evidence_status"], "found")
        self.assert_unverified(verdict)

    def test_disabled_with_only_one_success_returns_partial_opinion(self):
        panel = FakePanel(partial=True)
        verdict, queries = self.get_fact(httpx.Response(503), panel, provider="none")
        self.assertEqual(queries, [])
        self.assertEqual(verdict["evidence_status"], "disabled")
        self.assertEqual(verdict["discussion_status"], "partial")
        self.assert_unverified(verdict, discussed=1)

    def test_cited_agreement_preserves_normal_verdict_without_discussion(self):
        fixture = httpx.Response(200, json={"pages": [
            {"id": 1, "key": "Example", "title": "Example", "excerpt": "There were 200 registrations in 2025."}
        ]})
        verdict, _ = self.get_fact(fixture, FakePanel())
        self.assertEqual(verdict["stance"], "supported")
        self.assertEqual(verdict["cited_source_ids"], ["S1"])
        self.assertEqual(verdict["discussion"], [])
        self.assertEqual(verdict["discussion_status"], "not_needed")


    def test_all_yes_can_proceed_without_search_hits_but_not_verified(self):
        panel = FakePanel(votes=("yes", "yes", "yes"))
        verdict, _ = self.get_fact(httpx.Response(200, json={"pages": []}), panel)
        # 记忆模式下立场一致被接受，但摘要明确标注“未联网检索，置信度有限”。
        self.assertEqual(verdict["stance"], "supported")
        self.assertEqual(verdict["internal_decision"], "can_proceed")
        self.assertEqual([item["vote"] for item in verdict["internal_votes"]], ["yes"] * 3)
        self.assertTrue(all("internal_vote" in call[0]["content"] for call in panel.calls))

    def test_nonaffirmative_vote_or_missing_reviewer_does_not_approve(self):
        for vote in ("no", "uncertain", None, True, "maybe"):
            with self.subTest(vote=vote):
                panel = FakePanel(votes=("yes", "yes", vote))
                verdict, _ = self.get_fact(httpx.Response(200, json={"pages": []}), panel)
                self.assertEqual(verdict["internal_decision"], "no_unanimous_approval")
                self.assertEqual(verdict["internal_votes"][-1]["vote"], vote if vote in ("no", "uncertain") else "uncertain")
        panel = FakePanel(partial=True, votes=("yes", "yes", "yes"))
        verdict, _ = self.get_fact(httpx.Response(200, json={"pages": []}), panel)
        self.assertEqual(verdict["internal_decision"], "no_unanimous_approval")

    def test_external_refutation_is_not_hidden_by_internal_yes(self):
        fixture = httpx.Response(200, json={"pages": [
            {"id": 1, "key": "Example", "title": "Example", "excerpt": "There were 180 registrations in 2025."}
        ]})
        verdict, _ = self.get_fact(fixture, FakePanel(votes=("yes", "yes", "yes"), stance="refuted"))
        self.assertEqual(verdict["internal_decision"], "can_proceed")
        self.assertEqual(verdict["stance"], "refuted")
        self.assertEqual(verdict["cited_source_ids"], ["S1"])


    def test_middle_missing_is_located_without_renumbering_other_reviewers(self):
        panel = FakePanel(votes=("yes", "yes", "yes"), invalid_slot=2)
        verdict, _ = self.get_fact(httpx.Response(200, json={"pages": []}), panel)
        self.assertEqual(verdict["internal_votes"], [
            {"reviewer": "R1", "vote": "yes"}, {"reviewer": "R2", "vote": "missing"},
            {"reviewer": "R3", "vote": "yes"},
        ])
        # R2 缺失但其余评审立场一致：结论仍形成；因未全体同意，进入人工复核决定。
        self.assertEqual(verdict["stance"], "supported")
        self.assertEqual(verdict["discussion"], [])
        self.assertEqual(verdict["discussion_status"], "not_needed")
        self.assertEqual(verdict["internal_decision"], "no_unanimous_approval")

    def test_rate_limit_and_malformed_response_preserve_internal_yes(self):
        for response in (
            httpx.Response(429, headers={"Retry-After": "90"}),
            httpx.Response(200, content=b"invalid json"),
        ):
            with self.subTest(response=response):
                panel = FakePanel(votes=("yes", "yes", "yes"))
                verdict, queries = self.get_fact(response, panel)
                self.assertEqual(len(queries), 1)
                self.assertEqual(verdict["evidence_status"], "unavailable")
                self.assertEqual(verdict["internal_decision"], "can_proceed")
                # 检索不可用时进入记忆模式：立场一致仍形成结论。
                self.assertEqual(verdict["stance"], "supported")

    def test_real_search_hit_with_unknown_citation_is_not_confirmation(self):
        response = httpx.Response(200, json={"pages": [
            {"id": 1, "key": "Water", "title": "Water", "excerpt": "Water is a chemical substance."}
        ]})
        verdict, _ = self.get_fact(response, FakePanel(cited_id="S99", votes=("yes", "yes", "yes")))
        self.assertEqual(verdict["evidence_status"], "found")
        self.assertEqual(verdict["internal_decision"], "can_proceed")
        self.assert_unverified(verdict)

    def test_supporting_citation_never_overrides_dissenting_decision_vote(self):
        response = httpx.Response(200, json={"pages": [
            {"id": 1, "key": "Example", "title": "Example", "excerpt": "There were 200 registrations in 2025."}
        ]})
        verdict, _ = self.get_fact(response, FakePanel(votes=("yes", "yes", "no")))
        self.assertEqual(verdict["stance"], "supported")
        self.assertEqual(verdict["internal_decision"], "no_unanimous_approval")
        self.assertEqual(verdict["cited_source_ids"], ["S1"])
