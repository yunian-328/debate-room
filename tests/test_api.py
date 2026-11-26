from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.main import create_app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="debate-room-api-")
        self.settings = Settings(
            database_path=Path(self.temp.name) / "api.db",
            scenario_config=PROJECT_ROOT / "config" / "scenarios.yaml",
            github_webhook_secret="test-secret",
        )
        self.context = TestClient(create_app(self.settings))
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        self.temp.cleanup()

    def wait(self, task_id: str) -> dict:
        for _ in range(200):
            detail = self.client.get(f"/api/v1/tasks/{task_id}").json()
            if detail["status"] in {"done", "failed"}:
                return detail
            time.sleep(0.02)
        self.fail("任务超时未完成")

    def test_mock_workbench_and_manual_diff_without_webhook(self) -> None:
        page = self.client.get("/workbench")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn('endpoint: "/api/v1/hooks/code-review"', page.text)
        self.assertNotIn('endpoint: "/api/v1/hooks/github"', page.text)
        self.assertEqual(self.client.get("/").json()["mode"], "mock")
        path = "/api/v1/hooks/code-review"
        headers = {"Idempotency-Key": "workbench-manual-diff"}
        payload = {"content": "diff --git a/a.py b/a.py\n+print('demo')"}
        first = self.client.post(path, json=payload, headers=headers)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["scenario"], "code_review")
        duplicate = self.client.post(path, json=payload, headers=headers)
        self.assertEqual(duplicate.json()["task_id"], first.json()["task_id"])
        self.assertEqual(self.wait(first.json()["task_id"])["status"], "done")
        self.assertEqual(self.client.post(path, json={**payload, "callback_url": ""}).status_code, 400)

    def test_form_hook_idempotency_and_result(self) -> None:
        payload = {"content": "一条待评审的营销文案"}
        first = self.client.post(
            "/api/v1/hooks/form", json=payload, headers={"Idempotency-Key": "same-request"}
        )
        second = self.client.post(
            "/api/v1/hooks/form", json=payload, headers={"Idempotency-Key": "same-request"}
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["task_id"], second.json()["task_id"])
        self.assertTrue(first.json()["created"])
        self.assertFalse(second.json()["created"])
        detail = self.wait(first.json()["task_id"])
        self.assertEqual(detail["status"], "done")
        self.assertEqual(detail["scenario"], "copy_review")
        self.assertGreater(detail["total_cost_usd"], 0)
        stats = self.client.get("/api/v1/stats").json()
        self.assertEqual(stats["tasks"].get("done"), 1)

    def test_github_signature_and_code_review(self) -> None:
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 7,
                "html_url": "https://example.test/pr/7",
                "_links": {},
            },
            "repository": {"full_name": "owner/repo"},
            "diff": "diff --git a/a.py b/a.py\n+user_input = input()",
        }
        body = json.dumps(payload).encode()
        rejected = self.client.post(
            "/api/v1/hooks/github", content=body, headers={"X-GitHub-Event": "pull_request"}
        )
        self.assertEqual(rejected.status_code, 401)
        signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
        accepted = self.client.post(
            "/api/v1/hooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "delivery-7",
                "X-Hub-Signature-256": signature,
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(accepted.status_code, 202)
        detail = self.wait(accepted.json()["task_id"])
        self.assertEqual(detail["status"], "done")
        self.assertEqual(detail["payload"]["context"]["pr_number"], 7)

    def test_idempotency_conflict_does_not_return_wrong_task(self) -> None:
        headers = {"Idempotency-Key": "shared-key"}
        first = self.client.post("/api/v1/hooks/form", json={"content": "文案"}, headers=headers)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(self.client.post("/api/v1/hooks/fact-check", json={"content": "文案"}, headers=headers).status_code, 409)
        self.assertEqual(self.client.post("/api/v1/hooks/form", json={"content": "另一份"}, headers=headers).status_code, 409)

    def test_invalid_github_payloads_return_400(self) -> None:
        for payload in ([], {"action": "opened", "diff": 42}, {"action": "opened", "diff": " "}):
            body = json.dumps(payload).encode()
            signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            response = self.client.post("/api/v1/hooks/github", content=body,
                                        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": signature})
            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
