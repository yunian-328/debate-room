from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app, PREVIEW_MAX_BODY_BYTES, PREVIEW_MAX_SUBMISSIONS_PER_MINUTE
from app.openrouter import MockOpenRouterClient


class PrivatePreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="debate-room-preview-")
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            database_path=Path(self.temp.name) / "preview.db",
            mock_mode=False,
            preview_api_token="a-private-preview-test-token-only",
            openrouter_api_key="fake-model-credentials-not-used",
            github_webhook_secret="fake-unused-github-webhook-secret",
            github_token="fake-unused-github-token",
        )
        self.headers = {"Authorization": f"Bearer {self.settings.preview_api_token}"}
        self.enterContext(patch("app.main.OpenRouterClient", lambda _: MockOpenRouterClient()))
        self.notifier = self.enterContext(patch("app.main.ResultNotifier"))
        self.client = self.enterContext(TestClient(create_app(self.settings)))

    def test_real_mode_rejects_missing_short_or_shared_token(self) -> None:
        for token in ("", "short", self.settings.openrouter_api_key):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "DEBATE_ROOM_API_TOKEN"):
                create_app(replace(self.settings, preview_api_token=token))

    def test_task_and_docs_routes_require_private_token(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        # 静态页面公开可访问；数据接口与文档仍需令牌。
        for route in ("/", "/workbench"):
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 200)
        for route in ("/docs", "/openapi.json", "/api/v1/tasks", "/api/v1/tasks/x", "/api/v1/stats"):
            with self.subTest(route=route):
                unauthorized = self.client.get(route)
                self.assertEqual(unauthorized.status_code, 401)
                self.assertEqual(unauthorized.headers["cache-control"], "no-store")
        page = self.client.get("/workbench", headers=self.headers)
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        result = self.client.get("/api/v1/tasks", headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["cache-control"], "no-store")
        self.assertEqual(self.client.post("/api/v1/hooks/form", json={"content": "anything"}).status_code, 401)

    def test_preview_disables_callbacks_and_github_but_allows_manual_diff(self) -> None:
        callback = self.client.post("/api/v1/hooks/form", headers=self.headers,
                                    json={"content": "mock payload", "callback_url": "https://example.test/"})
        self.assertEqual(callback.status_code, 400)
        github = self.client.post("/api/v1/hooks/github", headers=self.headers,
                                  json={"action": "opened"})
        self.assertEqual(github.status_code, 404)
        manual = self.client.post("/api/v1/hooks/code-review", headers=self.headers,
                                  json={"content": "diff --git a/a.py b/a.py\n+print('demo')"})
        self.assertEqual(manual.status_code, 202)
        self.assertEqual(manual.json()["scenario"], "code_review")
        self.notifier.assert_not_called()

    def test_body_and_task_budget_are_bounded(self) -> None:
        route = "/api/v1/hooks/fact-check"
        result = self.client.post(route, headers=self.headers, content=b"x" * (PREVIEW_MAX_BODY_BYTES + 1))
        self.assertEqual(result.status_code, 413)
        result = self.client.post(route, headers=self.headers,
                                  content=iter([b"x" * (PREVIEW_MAX_BODY_BYTES // 2 + 1)] * 2))
        self.assertEqual(result.status_code, 413)
        self.assertEqual(self.client.post(route, headers=self.headers,
                                          json={"content": "x" * 8_001}).status_code, 413)
        for constraints in ({"max_cost_usd": 10}, {"max_latency_ms": 120_000}):
            self.assertEqual(self.client.post(route, headers=self.headers, json={
                "content": "synthetic claim", "constraints": constraints,
            }).status_code, 400)

    def test_preview_rate_limits_new_tasks(self) -> None:
        for index in range(PREVIEW_MAX_SUBMISSIONS_PER_MINUTE):
            reply = self.client.post("/api/v1/hooks/form", headers=self.headers,
                                     json={"content": f"synthetic text {index}"})
            self.assertEqual(reply.status_code, 202)
        reply = self.client.post("/api/v1/hooks/form", headers=self.headers, json={"content": "one more"})
        self.assertEqual(reply.status_code, 429)
