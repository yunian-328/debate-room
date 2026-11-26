"""Run one explicitly free synthetic chat completion. Never print credentials."""
from __future__ import annotations

import asyncio
import sys

from app.config import Settings
from app.errors import ReviewError
from app.models import ModelSpec
from app.openrouter import OpenRouterClient
from app.scenarios import FREE_PREVIEW_MODELS


async def main() -> int:
    try:
        client = OpenRouterClient(Settings.from_env())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        model = FREE_PREVIEW_MODELS[0]
        reply = await client.complete(
            ModelSpec(model, "google"),
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=24,
            timeout_seconds=15,
        )
        if not reply.content.strip() or reply.cost_usd != 0:
            print("Live check returned an empty response or a non-zero cost.", file=sys.stderr)
            return 1
        print(f"OpenRouter live OK · {reply.model} · cost USD 0")
        return 0
    except ReviewError as exc:
        print(f"OpenRouter live check failed: {exc.kind}", file=sys.stderr)
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
