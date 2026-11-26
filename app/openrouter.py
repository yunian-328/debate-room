from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Protocol

import httpx

from .config import Settings
from .errors import ErrorKind, ReviewError
from .models import ModelResponse, ModelSpec
from .scenarios import FREE_PREVIEW_MODELS


class LLMClient(Protocol):
    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ModelResponse: ...

    async def close(self) -> None: ...


class OpenRouterClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not settings.openrouter_api_key:
            raise ValueError("真实模式需要 OPENROUTER_API_KEY")
        self.custom = settings.openrouter_base_url != "https://openrouter.ai/api/v1"
        self.settings = settings
        self.http = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            transport=transport,
            trust_env=False,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "X-Title": "Debate Room",
            },
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        response = await self.http.get("/models", timeout=self.settings.request_timeout_seconds)
        response.raise_for_status()
        return response.json().get("data", [])

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ModelResponse:
        if not self.custom and spec.model not in FREE_PREVIEW_MODELS:
            raise ReviewError(ErrorKind.BAD_REQUEST, "预览仅允许明确配置的免费模型", model=spec.model)
        started = time.monotonic()
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if not self.custom:
            payload.update(
                {
                    "usage": {"include": True},
                    "plugins": [{"id": "web", "enabled": False}],
                    "provider": {"sort": "price", "allow_fallbacks": True},
                }
            )
        try:
            response = await self.http.post(
                "/chat/completions",
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ReviewError(
                ErrorKind.TIMEOUT, str(exc), model=spec.model
            ) from exc
        except httpx.HTTPError as exc:
            raise ReviewError(ErrorKind.PROVIDER, str(exc), model=spec.model) from exc

        if response.status_code >= 400:
            self._raise_http(response, spec.model)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ReviewError(
                ErrorKind.PROVIDER, "响应缺少 choices", model=spec.model
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or "stop"
        content = choice.get("message", {}).get("content") or ""
        if finish_reason == "content_filter":
            raise ReviewError(
                ErrorKind.CONTENT_FILTER, "模型触发内容过滤", model=spec.model
            )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        if cost is None:
            cost = (
                prompt_tokens * spec.input_per_million
                + completion_tokens * spec.output_per_million
            ) / 1_000_000
        return ModelResponse(
            model=data.get("model") or spec.model,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=float(cost),
            latency_ms=round((time.monotonic() - started) * 1000),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _raise_http(response: httpx.Response, model: str) -> None:
        status = response.status_code
        if status in {401, 402}:
            kind = ErrorKind.AUTH_BILLING
        elif status == 429:
            kind = ErrorKind.RATE_LIMIT
        elif status >= 500:
            kind = ErrorKind.PROVIDER
        else:
            kind = ErrorKind.BAD_REQUEST
        retry_after = None
        raw_retry = response.headers.get("Retry-After")
        if raw_retry:
            try:
                retry_after = float(raw_retry)
            except ValueError:
                pass
        raise ReviewError(
            kind,
            f"OpenRouter HTTP {status}",
            model=model,
            retry_after=retry_after,
        )


class MockOpenRouterClient:
    """离线确定性模型，用于演示和自动化测试。"""

    async def close(self) -> None:
        return None

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ModelResponse:
        await asyncio.sleep(0.01)
        started = time.monotonic()
        text = "\n".join(message["content"] for message in messages)
        content = self._answer(spec, text)
        prompt_tokens = max(1, len(text) // 3)
        completion_tokens = max(1, len(content) // 3)
        cost = (
            prompt_tokens * spec.input_per_million
            + completion_tokens * spec.output_per_million
        ) / 1_000_000
        return ModelResponse(
            model=spec.model,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=max(1, round((time.monotonic() - started) * 1000)),
        )

    @staticmethod
    def _digest(value: str) -> int:
        return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)

    def _answer(self, spec: ModelSpec, text: str) -> str:
        digest = self._digest(spec.model)
        score = 76 + digest % 19
        is_judge = "待汇总匿名评审" in text

        if "scenario=fact_check" in text:
            if "没有可用外部资料；这是内部辩论" in text and "触发分歧演示" not in text:
                viewpoints = (
                    ("可能支持理由", "如果原始定义与说法一致，结论才可能成立；需要查定义。"),
                    ("可能的反例", "不同样本或时间范围可能给出不同结果，不能直接外推。"),
                    ("最需要澄清", "先明确对象和时间范围，再寻找可以核对的原始材料。"),
                )
                answer = next((opinion for keyword, opinion in viewpoints if keyword in text), "需要查看原始材料。")
                return json.dumps(
                    {"answer": answer, "stance": "insufficient_evidence", "evidence_ids": [],
                     "internal_vote": "yes" if "触发一致同意演示" in text else "uncertain"},
                    ensure_ascii=False,
                )
            if "触发分歧演示" in text:
                answers = [
                    "现有证据支持该说法，但适用条件有限。",
                    "该说法与可靠证据相反，不应采信。",
                    "目前没有足够证据判断该说法。",
                ]
                stances = ["supported", "refuted", "insufficient_evidence"]
                # 按模型固定三种答案，保证该夹具始终覆盖“全不一致”路径。
                index = 0 if spec.model.startswith("openai/") else 1 if spec.model.startswith("anthropic/") else 2
                answer = answers[index]
                stance = stances[index]
            else:
                answer = "现有可验证证据不足，不建议直接采信该说法。"
                stance = "insufficient_evidence"
            return json.dumps(
                {"answer": answer, "stance": stance, "evidence_ids": [], "internal_vote": "uncertain"},
                ensure_ascii=False,
            )

        if "scenario=code_review" in text:
            if is_judge:
                return json.dumps(
                    {
                        "score": 82,
                        "summary": "综合评审发现一处需要确认的输入校验风险。",
                        "issues": [{
                            "title": "外部输入需要显式校验",
                            "severity": "medium",
                            "file": "diff 所示文件",
                            "line": 0,
                            "evidence": "变更引入了新的外部输入路径",
                            "suggestion": "在边界层增加类型与长度校验，并补充异常路径测试",
                            "confidence": "medium",
                        }],
                        "disagreements": [],
                        "confidence": "medium",
                        "needs_human": False,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "score": score,
                    "summary": f"{spec.role}完成审查。",
                    "issues": [{
                        "title": "外部输入需要显式校验",
                        "severity": "medium",
                        "file": "diff 所示文件",
                        "line": 0,
                        "evidence": "变更引入了新的外部输入路径",
                        "suggestion": "增加输入校验和异常路径测试",
                    }],
                },
                ensure_ascii=False,
            )

        if is_judge:
            scores = [int(value) for value in re.findall(r'"score"\s*:\s*(\d+)', text)]
            average = round(sum(scores) / len(scores)) if scores else 84
            return json.dumps(
                {
                    "score": average,
                    "summary": "文案核心信息清楚，建议收紧行动召唤并复核绝对化表述。",
                    "suggestions": ["将核心收益前置", "补充明确行动召唤", "复核无法证实的绝对化表述"],
                    "compliance_flags": [],
                    "disagreements": [],
                    "confidence": "high" if len(scores) >= 2 else "medium",
                    "needs_human": False,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "score": score,
                "summary": f"{spec.role}认为文案可读，但仍有优化空间。",
                "suggestions": [f"从{spec.role}角度补充一条可执行修改"],
                "compliance_flags": [],
            },
            ensure_ascii=False,
        )
