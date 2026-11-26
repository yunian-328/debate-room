from __future__ import annotations

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
from app.openrouter import MockOpenRouterClient


class WikimediaSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_hosts_bounded_query_sanitized_attributed_results(self):
        calls = []

        def fake(request):
            calls.append(request)
            self.assertEqual(request.url.path, "/w/rest.php/v1/search/page")
            self.assertEqual(request.url.params["limit"], "5")
            self.assertLessEqual(len(request.url.params["q"]), 600)
            self.assertIn("github.com/lennney/debate-room", request.headers["user-agent"])
            self.assertNotIn("authorization", request.headers)
            item = {"id": 4, "key": "A/B", "title": "Example", "excerpt": '<span class="searchmatch">A</span> &amp; B'}
            return httpx.Response(200, json={"pages": [item, item, {"id": 5, "key": "Empty", "title": "Empty", "excerpt": ""}]})

        for language, host in (("zh-HK", "zh.wikipedia.org"), ("en", "en.wikipedia.org")):
            sources = await WikimediaEvidenceSearch(transport=httpx.MockTransport(fake)).search("a" * 1200, language, timeout=1)
            self.assertEqual(calls[-1].url.host, host)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].id, "S1")
            self.assertEqual(sources[0].url, f"https://{host}/wiki/A%2FB")
            self.assertEqual(sources[0].excerpt, "A & B")
            self.assertIn("Wikipedia contributors", sources[0].attribution)
            self.assertTrue(sources[0].license_url.endswith("/by-sa/4.0/"))
        self.assertEqual(len(calls), 2)

    async def test_cooldown_honors_retry_after_without_second_request(self):
        for status in (429, 503):
            calls = []

            def fake(request):
                calls.append(request)
                return httpx.Response(status, headers={"Retry-After": "90"})

            search = WikimediaEvidenceSearch(transport=httpx.MockTransport(fake))
            with self.assertRaises(httpx.HTTPStatusError):
                await search.search("Synthetic claim", "en", timeout=1)
            self.assertGreater(search._next_request - time.monotonic(), 85)
            with self.assertRaisesRegex(ValueError, "cooldown"):
                await search.search("Another synthetic claim", "en", timeout=1)
            self.assertEqual(len(calls), 1)

    async def test_redirect_and_oversize_response_not_used(self):
        for response, error in [
            (httpx.Response(302, headers={"Location": "https://example.org"}), httpx.HTTPStatusError),
            (httpx.Response(200, content=b"x" * (128 * 1024 + 1)), ValueError),
        ]:
            with self.subTest(response=response):
                calls = []

                def fake(request):
                    calls.append(request)
                    return response

                search = WikimediaEvidenceSearch(transport=httpx.MockTransport(fake))
                with self.assertRaises(error):
                    await search.search("Synthetic", "en", timeout=1)
                self.assertEqual(len(calls), 1)


class WikimediaDefaultsTests(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(temp) / "test.db"
        self.token = "private-preview-test-token-only-123"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def get_client(self, **kwargs):
        client = TestClient(create_app(Settings(database_path=self.path, preview_api_token=self.token, **kwargs)))
        return self.enterContext(client)

    def wait(self, client, task):
        for _ in range(200):
            data = client.get(task, headers=self.headers).json()
            if data["status"] in {"failed", "done"}:
                return data
            time.sleep(0.01)
        self.fail("mock task did not finish")

    def test_mock_and_explicit_none_keep_the_adapter_disabled(self):
        creator = self.enterContext(patch("app.main.WikimediaEvidenceSearch", side_effect=AssertionError("don't instantiate for mock or explicit opt-out")))
        self.enterContext(patch("app.main.OpenRouterClient", return_value=MockOpenRouterClient()))
        for settings in ({}, {"mock_mode": False, "fact_search_provider": "none"}):
            client = self.get_client(**settings)
            self.assertEqual(client.get("/", headers=self.headers).json()["fact_search_provider"], "none")
        creator.assert_not_called()

    def test_default_real_settings_only_search_fact_check_and_abstain_on_empty(self):
        calls = []

        def fake(request):
            calls.append(request)
            return httpx.Response(200, json={"pages": []})

        search = WikimediaEvidenceSearch(transport=httpx.MockTransport(fake))
        self.enterContext(patch("app.main.WikimediaEvidenceSearch", return_value=search))
        self.enterContext(patch("app.main.OpenRouterClient", return_value=MockOpenRouterClient()))
        client = self.get_client(mock_mode=False)
        self.assertEqual(client.get("/", headers=self.headers).json()["fact_search_provider"], "wikimedia")
        for hook in ("form", "code-review", "fact-check"):
            task = client.post(f"/api/v1/hooks/{hook}", headers=self.headers, json={"content": "Synthetic item", "language": "en"})
            self.assertEqual(task.status_code, 202)
            detail = self.wait(client, task.json()["detail_url"])
            self.assertEqual(detail["status"], "done")
            if hook != "fact-check":
                self.assertEqual(calls, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(detail["verdict"]["stance"], "insufficient_evidence")
        self.assertEqual(detail["verdict"]["evidence_status"], "empty")
        self.assertEqual(detail["verdict"]["cited_source_ids"], [])

    def test_brave_remains_explicit_and_keyed(self):
        with self.assertRaisesRegex(ValueError, "BRAVE_SEARCH_API_KEY"):
            create_app(Settings(mock_mode=False, database_path=self.path, preview_api_token=self.token, fact_search_provider="brave"))
        brave = self.enterContext(patch("app.main.BraveEvidenceSearch"))
        self.enterContext(patch("app.main.OpenRouterClient", return_value=MockOpenRouterClient()))
        client = self.get_client(mock_mode=False, fact_search_provider="brave", brave_search_api_key="fixture-not-a-key")
        brave.assert_called_once_with("fixture-not-a-key")
        self.assertEqual(client.get("/", headers=self.headers).json()["fact_search_provider"], "brave")

    def test_http_success_with_uncited_snippet_does_not_prove_claim(self):
        payload = {"pages": [{"id": 1, "key": "Water", "title": "Water", "excerpt": "Water is a chemical substance."}]}
        search = WikimediaEvidenceSearch(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
        self.enterContext(patch("app.main.WikimediaEvidenceSearch", return_value=search))
        self.enterContext(patch("app.main.OpenRouterClient", return_value=MockOpenRouterClient()))
        client = self.get_client(mock_mode=False)
        submitted = client.post("/api/v1/hooks/fact-check", headers=self.headers, json={
            "content": "Synthetic claim with unrelated search result", "language": "en",
        })
        self.assertEqual(submitted.status_code, 202)
        detail = self.wait(client, submitted.json()["detail_url"])
        self.assertEqual(detail["verdict"]["evidence_status"], "found")
        self.assertEqual(detail["verdict"]["stance"], "insufficient_evidence")
        self.assertEqual(detail["verdict"]["cited_source_ids"], [])
