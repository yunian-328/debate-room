from __future__ import annotations

import asyncio
import time
from difflib import SequenceMatcher
from typing import Any

import httpx

from .evidence import EvidenceSearch, Source
from .errors import ErrorKind, ReviewError
from .models import ModelResponse, ModelSpec, Scenario
from .openrouter import LLMClient
from .prompts import extract_json, judge_messages, review_messages
from .router import ModelRouter
from .storage import Storage


class ReviewPipeline:
    def __init__(
        self,
        storage: Storage,
        router: ModelRouter,
        client: LLMClient,
        scenarios: dict[str, Scenario],
        *,
        max_retries_per_model: int = 1,
        default_timeout_seconds: float = 30.0,
        notifier: Any = None,
        evidence_search: EvidenceSearch | None = None,
    ) -> None:
        self.storage = storage
        self.router = router
        self.client = client
        self.scenarios = scenarios
        self.max_retries_per_model = max_retries_per_model
        self.default_timeout_seconds = default_timeout_seconds
        self.notifier = notifier
        self.evidence_search = evidence_search

    async def process(self, task_id: str) -> None:
        task = self.storage.get_task(task_id)
        if not task or task["status"] in {"done", "failed"}:
            return
        started = time.monotonic()
        try:
            self.storage.update_status(task_id, "routing")
            content = task["payload"]["content"]
            constraints = task["constraints"]
            sources: list[Source] = []
            evidence_status = "disabled"
            if task["scenario"] == "fact_check" and self.evidence_search:
                search_timeout = min(5.0, float(constraints["max_latency_ms"]) / 3000)
                try:
                    sources = await asyncio.wait_for(
                        self.evidence_search.search(
                            content, task["payload"]["language"], timeout=search_timeout
                        ), timeout=search_timeout
                    )
                    evidence_status = "found" if sources else "empty"
                except (ValueError, TimeoutError, httpx.HTTPError):
                    evidence_status = "unavailable"
            plan = self.router.make_plan(
                self.scenarios[task["scenario"]],
                content=content,
                max_cost_usd=float(constraints["max_cost_usd"]),
            )
            self.storage.update_status(task_id, "reviewing")
            calls = []
            configured_slots = {id(spec): index for index, spec in enumerate(plan.scenario.reviewers, start=1)}
            for index, chain in enumerate(plan.reviewer_chains, start=1):
                spec = chain[0]
                index = configured_slots.get(id(spec), index)
                messages = review_messages(
                    plan.scenario, spec, content, task["payload"]["language"],
                    sources=[source.as_dict() for source in sources],
                    reviewer_index=index,
                )
                calls.append(
                    self._run_chain(
                        task,
                        chain,
                        messages,
                        slot=f"reviewer_{index}",
                        max_tokens=plan.scenario.max_output_tokens,
                        started=started,
                    )
                )
            results = await asyncio.gather(*calls)
            reviews = [result for result in results if result is not None]
            minimum = 1
            if len(reviews) < minimum:
                raise ReviewError(
                    ErrorKind.PROVIDER,
                    f"成功评审数 {len(reviews)}，低于场景要求 {minimum}",
                )

            if plan.scenario.judge:
                self.storage.update_status(task_id, "judging")
                messages = judge_messages(
                    plan.scenario,
                    content,
                    reviews,
                    task["payload"]["language"],
                )
                judged = await self._run_chain(
                    task,
                    plan.judge_chain,
                    messages,
                    slot="judge",
                    # 裁判要汇总全部评审意见并输出完整 JSON，需比单个评审更长的输出预算，
                    # 否则 verbose 模型会截断输出导致 invalid_output。
                    max_tokens=plan.scenario.max_output_tokens + 1200,
                    started=started,
                )
                if judged is None:
                    raise ReviewError(ErrorKind.PROVIDER, "裁判及全部备选模型均失败")
                verdict = self._normalize_verdict(judged, reviews)
            else:
                verdict = self._cluster_fact_check(plan.scenario, reviews, sources, evidence_status)

            self.storage.save_verdict(task_id, verdict)
            self.storage.update_status(task_id, "done")
        except Exception as exc:
            kind = exc.kind if isinstance(exc, ReviewError) else ErrorKind.UNKNOWN
            self.storage.record_error(task_id, "", str(kind), 0, str(exc))
            self.storage.update_status(task_id, "failed", str(exc))
        finally:
            if self.notifier:
                await self.notifier.deliver(task_id)

    async def _run_chain(
        self,
        task: dict[str, Any],
        chain: list[ModelSpec],
        messages: list[dict[str, str]],
        *,
        slot: str,
        max_tokens: int,
        started: float,
    ) -> ModelResponse | None:
        task_id = task["task_id"]
        deadline_ms = int(task["constraints"]["max_latency_ms"])
        for spec in chain:
            for attempt in range(self.max_retries_per_model + 1):
                elapsed_ms = (time.monotonic() - started) * 1000
                remaining_seconds = (deadline_ms / 1000) - (time.monotonic() - started)
                if remaining_seconds <= 0:
                    self.storage.record_error(
                        task_id, spec.model, ErrorKind.DEADLINE, attempt, "任务总延迟预算耗尽"
                    )
                    return None
                try:
                    response = await self.client.complete(
                        spec,
                        messages,
                        max_tokens=max_tokens,
                        timeout_seconds=min(self.default_timeout_seconds, remaining_seconds),
                    )
                    response.slot = slot
                    response.role = spec.role
                    response.requested_model = spec.model
                    response.parsed = extract_json(response.content)
                    parsed = response.parsed
                    if task["scenario"] == "fact_check":
                        valid = isinstance(parsed.get("answer"), str) and bool(parsed["answer"].strip()) and parsed.get("stance") in ("supported", "refuted", "insufficient_evidence")
                    else:
                        score = parsed.get("score")
                        valid = (isinstance(parsed.get("summary"), str) and bool(parsed["summary"].strip())
                                 and isinstance(score, (int, float)) and not isinstance(score, bool)
                                 and 0 <= score <= 100)
                    if not valid:
                        self.storage.record_output(task_id, response, status="invalid_output")
                        self.storage.record_error(task_id, spec.model, "invalid_output", attempt, "模型输出缺少有效必填字段")
                        break
                    self.storage.record_output(task_id, response)
                    return response
                except ReviewError as exc:
                    self._record_failure(task_id, spec, slot, exc, attempt, started)
                    if exc.fatal:
                        raise
                    should_retry = exc.retryable and attempt < self.max_retries_per_model
                    if should_retry:
                        delay = exc.retry_after or min(2**attempt, 8)
                        await asyncio.sleep(min(delay, max(0.0, remaining_seconds)))
                        continue
                    # 内容拒绝、请求错误和重试耗尽均切换至备选模型。
                    break
        return None

    def _record_failure(
        self,
        task_id: str,
        spec: ModelSpec,
        slot: str,
        exc: ReviewError,
        attempt: int,
        started: float,
    ) -> None:
        response = ModelResponse(
            model=spec.model,
            content="",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0,
            latency_ms=round((time.monotonic() - started) * 1000),
            finish_reason="error",
            slot=slot,
            role=spec.role,
        )
        self.storage.record_output(task_id, response, status=str(exc.kind))
        self.storage.record_error(task_id, spec.model, str(exc.kind), attempt, exc.detail)

    @staticmethod
    def _normalize_verdict(
        judged: ModelResponse, reviews: list[ModelResponse]
    ) -> dict[str, Any]:
        verdict = dict(judged.parsed or extract_json(judged.content))
        if "raw_output" in verdict:
            verdict["summary"] = verdict.pop("raw_output")
        scores = [
            review.parsed.get("score")
            for review in reviews
            if isinstance(review.parsed.get("score"), (int, float))
        ]
        verdict.setdefault("score", round(sum(scores) / len(scores), 1) if scores else None)
        verdict.setdefault("summary", "裁判模型未提供摘要")
        verdict.setdefault("disagreements", [])
        verdict.setdefault("confidence", "medium")
        verdict.setdefault("needs_human", False)
        verdict["judge_model"] = judged.model
        return verdict

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        normalized = "".join(text.lower().split())
        if len(normalized) < 2:
            return {normalized}
        return {normalized[i : i + 2] for i in range(len(normalized) - 1)}

    @classmethod
    def _similarity(cls, left: str, right: str) -> float:
        a, b = cls._bigrams(left), cls._bigrams(right)
        if not a or not b:
            return SequenceMatcher(None, left, right).ratio()
        dice = 2 * len(a & b) / (len(a) + len(b))
        sequence = SequenceMatcher(None, left, right).ratio()
        return 0.7 * dice + 0.3 * sequence

    def _cluster_fact_check(
        self, scenario: Scenario, reviews: list[ModelResponse],
        sources: list[Source] | None = None, evidence_status: str = "disabled",
    ) -> dict[str, Any]:
        answers = [
            str(review.parsed.get("answer") or review.content).strip()
            for review in reviews
        ]
        valid_stances = {"supported", "refuted", "insufficient_evidence"}
        stances = [review.parsed.get("stance") for review in reviews]
        stances = [stance if isinstance(stance, str) and stance in valid_stances else None for stance in stances]
        unused = set(range(len(reviews)))
        evidence_backed = bool(sources)  # 有检索来源时按来源取证；无来源（记忆模式）仅凭立场一致性
        clusters: list[list[int]] = []
        while unused:
            seed = min(unused)
            unused.remove(seed)
            cluster = [seed]
            for other in sorted(list(unused)):
                if (
                    stances[seed] in valid_stances
                    and stances[seed] == stances[other]
                    and (not evidence_backed or self._similarity(answers[seed], answers[other]) >= scenario.similarity_threshold)
                ):
                    cluster.append(other)
                    unused.remove(other)
            clusters.append(cluster)
        largest = max(clusters, key=len)
        representative = answers[largest[0]]
        has_invalid_stance = any(stance not in valid_stances for stance in stances)
        has_conflicting_stance = len(set(stance for stance in stances if stance in valid_stances)) > 1
        source_by_id = {source.id: source for source in (sources or [])}
        stance = stances[largest[0]]
        shared_ids: set[str] = set(source_by_id)
        for index in largest:
            cited = reviews[index].parsed.get("evidence_ids")
            cited_ids = {value for value in cited if isinstance(value, str)} if isinstance(cited, list) else set()
            shared_ids &= cited_ids
        accepted = (
            stance in {"supported", "refuted"}
            and len({reviews[index].model.split("/", 1)[0] for index in largest}) >= scenario.min_agreement
            and not has_invalid_stance and not has_conflicting_stance
            and (bool(shared_ids) if evidence_backed else True)
        )
        agreement = len(largest) / len(reviews)
        expected = {f"reviewer_{index}": spec.model for index, spec in enumerate(scenario.reviewers, start=1)}
        returned = {review.slot: review for review in reviews}
        votes = []
        for slot, model in expected.items():
            row = returned.get(slot)
            if row is None or row.requested_model != model:
                vote = "missing"
            else:
                vote = row.parsed.get("internal_vote")
                if vote not in ("yes", "no", "uncertain"):
                    vote = "uncertain"
            votes.append({"reviewer": "R" + slot.removeprefix("reviewer_"), "vote": vote})
        unanimous = (
            len(votes) >= 2 and len(reviews) == len(expected)
            and len({spec.vendor for spec in scenario.reviewers}) >= 2
            and all(item["vote"] == "yes" for item in votes)
        )
        reviewer_labels = [
            "R" + review.slot.removeprefix("reviewer_")
            if review.slot.startswith("reviewer_") else f"R{index + 1}"
            for index, review in enumerate(reviews)
        ]
        discussion = [
            {"reviewer": reviewer_labels[index], "opinion": answer}
            for index, answer in enumerate(answers)
        ] if not accepted else []
        alternatives = [
            {"reviewer": reviewer_labels[index], "answer": answers[index]}
            for index in range(len(reviews))
            if index not in largest
        ]
        return {
            "judge_model": "bigram-cluster",
            "score": round(agreement * 100, 1),
            "summary": (
                f"{len(largest)}/{len(reviews)} 个评审在同一检索来源上形成一致结论；请核对原文。"
                if accepted and evidence_backed
                else f"{len(largest)}/{len(reviews)} 个评审立场一致（未联网检索，凭模型知识作答，结论置信度有限）。"
                if accepted
                else "证据不足，不能形成事实核查结论。下面是未证实的内部意见，需人工复核。"
            ),
            "answer": representative if accepted else None,
            "stance": stance if accepted else "insufficient_evidence",
            "evidence_status": evidence_status,
            "sources": [source.as_dict() for source in source_by_id.values()],
            "cited_source_ids": sorted(shared_ids) if accepted else [],
            "alternatives": alternatives,
            "disagreements": [item["answer"] for item in alternatives],
            "discussion": discussion,
            "internal_decision": "can_proceed" if unanimous else "no_unanimous_approval",
            "internal_votes": votes,
            "discussion_status": (
                "partial" if any(item["vote"] == "missing" for item in votes) else "unverified"
            ) if discussion else "not_needed",
            "confidence": "medium" if accepted else "low",
            "needs_human": not accepted,
            "clusters": [[reviewer_labels[index] for index in cluster] for cluster in clusters],
        }
