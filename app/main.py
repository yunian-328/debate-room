from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from starlette.datastructures import Headers
from starlette.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings
from .evidence import BraveEvidenceSearch, WikimediaEvidenceSearch
from .models import Submission
from .notifier import ResultNotifier
from .openrouter import MockOpenRouterClient, OpenRouterClient
from .pipeline import ReviewPipeline
from .router import ModelRouter
from .scenarios import load_scenarios, use_custom_models, use_free_preview_models
from .storage import Storage


PREVIEW_MAX_BODY_BYTES = 65_536
PREVIEW_MAX_CONTENT_CHARS = 8_000
PREVIEW_MAX_WAITING_TASKS = 4
PREVIEW_MAX_SUBMISSIONS_PER_MINUTE = 4
PREVIEW_MAX_SUBMISSIONS_PER_DAY = 24
PREVIEW_MAX_ESTIMATED_COST_USD = 0.10
PREVIEW_MAX_LATENCY_MS = 60_000


class PrivatePreviewMiddleware:
    def __init__(self, app: Any, *, token: str) -> None:
        self.app = app
        self.expected = b"Bearer " + token.encode("utf-8")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)

        async def private_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() != b"cache-control"
                ] + [(b"cache-control", b"no-store")]}
            await send(message)

        async def reject(code: int, detail: str, extra: dict | None = None) -> None:
            await JSONResponse({"detail": detail}, status_code=code, headers=extra)(
                scope, receive, private_send
            )

        public_health = scope["path"] == "/health" and scope["method"] == "GET"
        # 静态工作台页面本身不含数据，与 API 一样本机可用：页面公开，数据接口仍需令牌。
        public_page = scope["method"] == "GET" and scope["path"] in ("/", "/workbench")
        supplied = headers.getlist("authorization")
        if not (public_health or public_page) and (
            len(supplied) != 1
            or not hmac.compare_digest(supplied[0].encode("latin-1"), self.expected)
        ):
            await reject(401, "需要 Debate Room 访问令牌", {"WWW-Authenticate": "Bearer"})
            return

        if scope["method"] != "POST":
            await self.app(scope, receive, private_send)
            return

        content_length = headers.get("content-length")
        if content_length is not None and (
            not content_length.isascii()
            or not content_length.isdigit()
            or len(content_length) > len(str(PREVIEW_MAX_BODY_BYTES))
            or int(content_length) > PREVIEW_MAX_BODY_BYTES
        ):
            await reject(413, "请求体超出预览限制")
            return

        body: bytearray | None = bytearray()
        while True:
            event = await receive()
            if event["type"] != "http.request":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > PREVIEW_MAX_BODY_BYTES:
                await reject(413, "请求体超出预览限制")
                return
            body.extend(chunk)
            if not event.get("more_body", False):
                break

        async def replay() -> dict:
            nonlocal body
            if body is None:
                return await receive()
            data = bytes(body)
            body = None
            return {"type": "http.request", "body": data, "more_body": False}

        await self.app(scope, replay, private_send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    private_preview = not settings.mock_mode
    if private_preview:
        token = settings.preview_api_token
        if (
            not 32 <= len(token) <= 200
            or any(not 33 <= ord(char) <= 126 for char in token)
            or token == settings.openrouter_api_key
        ):
            raise ValueError("真实模式需要独立的 DEBATE_ROOM_API_TOKEN（32–200 个可打印 ASCII 字符）")
    evidence_search = None
    provider = settings.fact_search_provider or "wikimedia"
    if provider not in {"wikimedia", "brave", "none"}:
        raise ValueError("FACT_SEARCH_PROVIDER 必须是 wikimedia、brave 或 none")
    if provider == "brave":
        if not private_preview or not settings.brave_search_api_key:
            raise ValueError("Brave 搜索需真实模式及 BRAVE_SEARCH_API_KEY")
        evidence_search = BraveEvidenceSearch(settings.brave_search_api_key)
    elif provider == "wikimedia" and private_preview:
        evidence_search = WikimediaEvidenceSearch(contact_url=settings.wikimedia_contact_url)
    scenarios = load_scenarios(settings.scenario_config)
    if private_preview:
        custom_base = settings.openrouter_base_url != "https://openrouter.ai/api/v1"
        custom_models = [m.strip() for m in settings.custom_models.split(",") if m.strip()]
        if custom_base:
            scenarios = use_custom_models(scenarios, custom_models or ["deepseek-chat"])
        else:
            scenarios = use_free_preview_models(scenarios)
    storage = Storage(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = (
            MockOpenRouterClient()
            if settings.mock_mode
            else OpenRouterClient(settings)
        )
        router = ModelRouter(storage)
        notifier = None if private_preview else ResultNotifier(storage, settings)
        pipeline = ReviewPipeline(
            storage,
            router,
            client,
            scenarios,
            max_retries_per_model=settings.max_retries_per_model,
            default_timeout_seconds=settings.request_timeout_seconds,
            notifier=notifier,
            evidence_search=evidence_search,
        )
        queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=PREVIEW_MAX_WAITING_TASKS if private_preview else 0
        )
        app.state.pipeline = pipeline
        app.state.queue = queue

        async def worker() -> None:
            while True:
                task_id = await queue.get()
                try:
                    await pipeline.process(task_id)
                finally:
                    queue.task_done()

        worker_task = asyncio.create_task(worker(), name="review-worker")
        for task_id in storage.unfinished_task_ids():
            if private_preview:
                storage.update_status(task_id, "failed", "预览重启，未自动重发此前的付费任务")
            else:
                queue.put_nowait(task_id)
        try:
            yield
        finally:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
            await client.close()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="多模型交叉评审：文案质检、代码评审、事实核查。",
        lifespan=lifespan,
    )
    if private_preview:
        app.add_middleware(PrivatePreviewMiddleware, token=settings.preview_api_token)
    submissions: deque[float] = deque()
    app.state.settings = settings
    app.state.storage = storage
    app.state.scenarios = scenarios

    def submit(
        scenario: str,
        submission: Submission,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        if private_preview:
            if submission.callback_url is not None:
                raise HTTPException(status_code=400, detail="真实预览不支持外部回调")
            if len(submission.content) > PREVIEW_MAX_CONTENT_CHARS:
                raise HTTPException(status_code=413, detail="评审文本超出预览限制")
            if (
                submission.constraints.max_cost_usd > PREVIEW_MAX_ESTIMATED_COST_USD
                or submission.constraints.max_latency_ms > PREVIEW_MAX_LATENCY_MS
                or (idempotency_key is not None and len(idempotency_key) > 128)
            ):
                raise HTTPException(status_code=400, detail="任务参数超出预览限制")
            now = time.monotonic()
            while submissions and now - submissions[0] >= 86_400:
                submissions.popleft()
            if (
                app.state.queue.full()
                or len(submissions) >= PREVIEW_MAX_SUBMISSIONS_PER_DAY
                or sum(now - started < 60 for started in submissions) >= PREVIEW_MAX_SUBMISSIONS_PER_MINUTE
            ):
                raise HTTPException(status_code=429, detail="预览队列或提交频率已达到上限")
        try:
            task_id, created = storage.create_task(
                scenario, submission, idempotency_key=idempotency_key
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if created:
            if private_preview:
                submissions.append(now)
            app.state.queue.put_nowait(task_id)
        return {
            "task_id": task_id,
            "scenario": scenario,
            "status": storage.get_task(task_id)["status"],
            "created": created,
            "detail_url": f"/api/v1/tasks/{task_id}",
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "mode": "mock" if settings.mock_mode else "openrouter",
            "docs": "/docs",
            "fact_search_provider": provider if evidence_search else "none",
            "scenarios": {
                name: scenario.description for name, scenario in scenarios.items()
            },
        }

    @app.get("/workbench", include_in_schema=False)
    async def workbench() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "mock" if settings.mock_mode else "openrouter"}

    @app.post("/api/v1/hooks/form", status_code=status.HTTP_202_ACCEPTED)
    async def form_hook(
        submission: Submission,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return submit("copy_review", submission, idempotency_key)

    @app.post("/api/v1/hooks/fact-check", status_code=status.HTTP_202_ACCEPTED)
    async def fact_check_hook(
        submission: Submission,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return submit("fact_check", submission, idempotency_key)

    @app.post("/api/v1/hooks/code-review", status_code=status.HTTP_202_ACCEPTED)
    async def code_review_hook(
        submission: Submission,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        if submission.callback_url is not None or "comments_url" in submission.context:
            raise HTTPException(status_code=400, detail="手动评审不支持外部回调或 PR 评论")
        return submit("code_review", submission, idempotency_key)

    @app.post("/api/v1/hooks/github", status_code=status.HTTP_202_ACCEPTED)
    async def github_hook(request: Request) -> dict[str, Any]:
        if private_preview:
            raise HTTPException(status_code=404, detail="真实预览未启用 GitHub Webhook")
        body = await request.body()
        if settings.github_webhook_secret:
            supplied = request.headers.get("X-Hub-Signature-256", "")
            expected = "sha256=" + hmac.new(
                settings.github_webhook_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(supplied, expected):
                raise HTTPException(status_code=401, detail="GitHub webhook 签名校验失败")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="无效 JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Webhook 内容必须是 JSON 对象")

        event = request.headers.get("X-GitHub-Event", "")
        if event == "ping":
            return {"status": "pong"}
        if event != "pull_request":
            raise HTTPException(status_code=400, detail="仅支持 pull_request 事件")
        action = payload.get("action")
        if action not in {"opened", "synchronize", "reopened", "demo"}:
            return {"status": "ignored", "action": action}

        pull_request = payload.get("pull_request") or {}
        if not isinstance(pull_request, dict):
            raise HTTPException(status_code=400, detail="pull_request 必须是对象")
        diff = payload.get("diff") or pull_request.get("diff")
        if diff is not None and not isinstance(diff, str):
            raise HTTPException(status_code=400, detail="diff 必须是文本")
        if not diff:
            diff_url = pull_request.get("diff_url")
            if not diff_url:
                raise HTTPException(status_code=400, detail="缺少 PR diff_url")
            headers = {"Accept": "application/vnd.github.v3.diff"}
            if settings.github_token:
                headers["Authorization"] = f"Bearer {settings.github_token}"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    diff_url, headers=headers, timeout=settings.request_timeout_seconds
                )
                response.raise_for_status()
                diff = response.text

        if not diff.strip():
            raise HTTPException(status_code=400, detail="diff 不能为空")

        truncated = len(diff) > 180_000
        if truncated:
            diff = diff[:180_000] + "\n\n[系统提示：diff 过长，已截断，结论需人工复核]"
        comments_url = (pull_request.get("_links") or {}).get("comments", {}).get("href")
        submission = Submission(
            content=diff,
            language="zh",
            context={
                "repository": (payload.get("repository") or {}).get("full_name"),
                "pr_number": pull_request.get("number"),
                "pr_url": pull_request.get("html_url"),
                "comments_url": comments_url,
                "truncated": truncated,
            },
        )
        return submit(
            "code_review", submission, request.headers.get("X-GitHub-Delivery")
        )

    @app.get("/api/v1/tasks")
    async def list_tasks(limit: int = 50) -> list[dict[str, Any]]:
        return storage.list_tasks(min(max(limit, 1), 200))

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        detail = storage.get_task_detail(task_id)
        if not detail:
            raise HTTPException(status_code=404, detail="任务不存在")
        return detail

    @app.get("/api/v1/stats")
    async def stats() -> dict[str, Any]:
        return storage.stats()

    return app


app = create_app()
