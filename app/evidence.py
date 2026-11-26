"""Search first, then give the same source excerpts to every reviewer."""
from __future__ import annotations

import asyncio
import html
import json
import time
import re
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import quote, urlsplit

import httpx


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    title: str
    excerpt: str
    attribution: str | None = None
    license_url: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class EvidenceSearch(Protocol):
    async def search(self, claim: str, language: str, *, timeout: float) -> list[Source]: ...


_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _plain(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]*>", "", value)).split())[:limit]


class BraveEvidenceSearch:
    """One bounded lookup. Result URLs are displayed, never fetched by the app."""

    def __init__(self, token: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._token = token
        self._transport = transport

    async def search(self, claim: str, language: str, *, timeout: float) -> list[Source]:
        params: dict[str, str | int] = {
            "q": " ".join(claim.split()[:75])[:600],
            "count": 5,
            "text_decorations": "false",
            "result_filter": "web",
        }
        normalized = language.lower().replace("_", "-")
        if normalized in {"zh-tw", "zh-hk", "zh-hant"}:
            params["search_lang"] = "zh-hant"
        elif normalized.startswith("zh"):
            params["search_lang"] = "zh-hans"
        async with httpx.AsyncClient(transport=self._transport, follow_redirects=False, trust_env=False) as client:
            resp = await client.get(
                _BRAVE_URL, params=params, timeout=timeout,
                headers={"X-Subscription-Token": self._token, "Accept": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
        if not isinstance(result, dict):
            raise ValueError("Unexpected search response")
        web = result.get("web", {})
        rows = web.get("results", []) if isinstance(web, dict) else None
        if not isinstance(rows, list):
            raise ValueError("Unexpected search results")
        sources: list[Source] = []
        seen: set[str] = set()
        for item in rows[:5]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url):
                continue
            try:
                location = urlsplit(url)
                if location.scheme != "https" or not location.hostname or location.username is not None:
                    continue
                normalized_url = location._replace(fragment="").geturl()
            except ValueError:
                continue
            excerpt = _plain(item.get("description"), 600)
            if normalized_url in seen or not excerpt:
                continue
            seen.add(normalized_url)
            sources.append(Source(f"S{len(sources) + 1}", url, _plain(item.get("title"), 160), excerpt))
        return sources


_WIKI_HOSTS = {"en": "en.wikipedia.org", "zh": "zh.wikipedia.org"}
_WIKI_DEFAULT_CONTACT = "https://github.com/lennney/debate-room"
_WIKI_LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
_WIKI_RESPONSE_LIMIT = 128 * 1024


def _retry_after_seconds(value: str | None) -> float:
    if value:
        if value.isascii() and value.isdigit() and len(value) <= 9:
            return max(1.0, float(value))
        try:
            stamp = parsedate_to_datetime(value)
            if stamp.tzinfo is not None:
                return max(1.0, stamp.timestamp() - time.time())
        except (ValueError, TypeError, OverflowError, OSError):
            pass
    return 60.0


class WikimediaEvidenceSearch:
    """Public Wikipedia search, not a general web index or a primary source."""

    def __init__(
        self, *, transport: httpx.AsyncBaseTransport | None = None, contact_url: str = ""
    ):
        self._transport = transport
        contact_url = contact_url or _WIKI_DEFAULT_CONTACT
        contact = urlsplit(contact_url)
        if (
            contact.scheme != "https" or not contact.hostname or contact.username is not None
            or len(contact_url) > 250 or any(not 33 <= ord(char) <= 126 or char in "()" for char in contact_url)
        ):
            raise ValueError("WIKIMEDIA_CONTACT_URL must be a public HTTPS project contact URL")
        # Identification only; this URL is never requested by this app.
        self._user_agent = f"DebateRoom/0.1 ({contact_url})"
        self._inflight = asyncio.Lock()
        self._next_request = 0.0

    async def search(self, claim: str, language: str, *, timeout: float) -> list[Source]:
        query = " ".join(claim.split()[:75])[:600]
        if not query:
            return []
        host = _WIKI_HOSTS["zh" if language.lower().replace("_", "-").split("-")[0] == "zh" else "en"]
        url = f"https://{host}/w/rest.php/v1/search/page"
        if self._inflight.locked():
            raise ValueError("Wikimedia search busy")
        async with self._inflight:
            now = time.monotonic()
            if now < self._next_request:
                raise ValueError("Wikimedia search cooldown")
            # Never retry. Preemptively separate requests if a page returns quickly.
            self._next_request = now + 1.0
            async with httpx.AsyncClient(transport=self._transport, follow_redirects=False, trust_env=False) as client:
                async with client.stream(
                    "GET", url, params={"q": query, "limit": 5}, timeout=timeout,
                    headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                ) as response:
                    if response.status_code in {429, 503}:
                        self._next_request = max(
                            self._next_request, time.monotonic() +
                            _retry_after_seconds(response.headers.get("Retry-After")),
                        )
                    response.raise_for_status()
                    body = bytearray()
                    async for piece in response.aiter_bytes():
                        if len(body) + len(piece) > _WIKI_RESPONSE_LIMIT:
                            raise ValueError("Wikimedia response too large")
                        body.extend(piece)
                    raw = json.loads(body)
        pages = raw.get("pages") if isinstance(raw, dict) else None
        if not isinstance(pages, list):
            raise ValueError("Unexpected Wikimedia response")
        sources: list[Source] = []
        seen: set[int] = set()
        for item in pages[:5]:
            if not isinstance(item, dict):
                continue
            page_id = item.get("id")
            key = item.get("key")
            excerpt = _plain(item.get("excerpt"), 600)
            title = _plain(item.get("title"), 160)
            if (
                type(page_id) is not int or page_id <= 0 or page_id in seen
                or not isinstance(key, str) or not 1 <= len(key) <= 255 or key in {".", ".."}
                or not title or not excerpt
            ):
                continue
            seen.add(page_id)
            sources.append(Source(
                f"S{len(sources) + 1}", f"https://{host}/wiki/{quote(key, safe='')}",
                title, excerpt, "Wikipedia contributors (snippet; markup removed and text may be shortened)",
                _WIKI_LICENSE,
            ))
        return sources
