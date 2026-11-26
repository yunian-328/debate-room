from __future__ import annotations

import json

import httpx

from .config import Settings
from .errors import ErrorKind
from .storage import Storage


class ResultNotifier:
    """通用 HTTP 回调；配置 GitHub Token 时也可回发 PR 评论。"""

    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    async def deliver(self, task_id: str) -> None:
        detail = self.storage.get_task_detail(task_id)
        if not detail:
            return
        callback_url = detail.get("callback_url")
        if callback_url and callback_url.startswith(("http://", "https://")):
            await self._post_callback(task_id, callback_url, detail)

        context = detail["payload"].get("context", {})
        comments_url = context.get("comments_url")
        if (
            detail["scenario"] == "code_review"
            and detail["status"] == "done"
            and comments_url
            and self.settings.github_token
        ):
            await self._post_github_comment(task_id, comments_url, detail)

    async def _post_callback(
        self, task_id: str, url: str, detail: dict
    ) -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, json=detail, timeout=self.settings.callback_timeout_seconds
                )
                response.raise_for_status()
        except Exception as exc:
            self.storage.record_error(
                task_id, "", ErrorKind.CALLBACK, 0, f"结果回调失败：{exc}"
            )

    async def _post_github_comment(
        self, task_id: str, url: str, detail: dict
    ) -> None:
        verdict = detail.get("verdict") or {}
        issues = verdict.get("issues") or []
        lines = [
            "## 多模型交叉代码评审",
            "",
            f"**综合评分：** {verdict.get('score', '—')} / 100  ",
            f"**置信度：** {verdict.get('confidence', '—')}  ",
            f"**任务 ID：** `{task_id}`",
            "",
            verdict.get("summary", ""),
            "",
        ]
        for issue in issues[:10]:
            lines.extend(
                [
                    f"### {issue.get('severity', 'medium')} · {issue.get('title', '未命名问题')}",
                    f"- 位置：`{issue.get('file', '—')}:{issue.get('line', 0)}`",
                    f"- 证据：{issue.get('evidence', '—')}",
                    f"- 建议：{issue.get('suggestion', '—')}",
                    f"- 置信度：{issue.get('confidence', 'medium')}",
                    "",
                ]
            )
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.settings.github_token}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json={"body": "\n".join(lines)},
                    timeout=self.settings.callback_timeout_seconds,
                )
                response.raise_for_status()
        except Exception as exc:
            self.storage.record_error(
                task_id, "", ErrorKind.CALLBACK, 0, f"GitHub 评论回发失败：{exc}"
            )
