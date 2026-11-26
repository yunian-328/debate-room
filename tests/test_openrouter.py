import json
import unittest

import httpx

from app.config import Settings
from app.errors import ReviewError
from app.models import ModelSpec
from app.openrouter import OpenRouterClient
from app.scenarios import FREE_PREVIEW_MODELS


class OpenRouterPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_path_uses_free_slug_and_no_search(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], FREE_PREVIEW_MODELS[0])
            self.assertEqual(payload["plugins"], [{"id": "web", "enabled": False}])
            self.assertNotIn("tools", payload)
            return httpx.Response(200, json={
                "model": FREE_PREVIEW_MODELS[0],
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1, "cost": 0},
            })
        client = OpenRouterClient(Settings(openrouter_api_key="test-key"), transport=httpx.MockTransport(handler))
        try:
            reply = await client.complete(ModelSpec(FREE_PREVIEW_MODELS[0], "google"), [], max_tokens=8, timeout_seconds=1)
            self.assertEqual(reply.content, "OK")
            self.assertEqual(reply.cost_usd, 0)
            with self.assertRaises(ReviewError):
                await client.complete(ModelSpec("openai/gpt-4o", "openai"), [], max_tokens=8, timeout_seconds=1)
        finally:
            await client.close()

    def test_credentials_not_sent_to_custom_url(self):
        with self.assertRaises(ValueError):
            OpenRouterClient(Settings(openrouter_api_key="test-key", openrouter_base_url="https://example.org"))
