from __future__ import annotations

from .errors import ErrorKind, ReviewError
from .models import ModelSpec, ReviewPlan, Scenario
from .storage import Storage


class ModelRouter:
    """按预算和近期健康度选模，并为每个槽位生成独立降级链。"""

    def __init__(self, storage: Storage, unhealthy_error_rate: float = 0.75) -> None:
        self.storage = storage
        self.unhealthy_error_rate = unhealthy_error_rate

    def is_healthy(self, spec: ModelSpec) -> bool:
        health = self.storage.model_health(spec.model)
        return not (
            health["attempts"] >= 4
            and health["error_rate"] >= self.unhealthy_error_rate
        )

    @staticmethod
    def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * spec.input_per_million
            + output_tokens * spec.output_per_million
        ) / 1_000_000

    def _chain(self, primary: ModelSpec, fallbacks: list[ModelSpec]) -> list[ModelSpec]:
        chain = [primary]
        seen = {primary.model}
        for spec in fallbacks:
            if spec.model not in seen and self.is_healthy(spec):
                chain.append(spec)
                seen.add(spec.model)
        return chain

    def make_plan(
        self,
        scenario: Scenario,
        *,
        content: str,
        max_cost_usd: float,
    ) -> ReviewPlan:
        input_tokens = max(100, len(content) // 3)
        remaining = max_cost_usd
        selected: list[ModelSpec] = []

        for spec in scenario.reviewers:
            if not self.is_healthy(spec):
                continue
            estimate = self.estimate_cost(spec, input_tokens, scenario.max_output_tokens)
            if estimate <= remaining:
                selected.append(spec)
                remaining -= estimate

        if len(selected) < 2:
            raise ReviewError(
                ErrorKind.BUDGET,
                f"预算 ${max_cost_usd:.4f} 不足以选择至少两个健康评审模型",
            )
        if len({spec.vendor for spec in selected}) < 2:
            raise ReviewError(
                ErrorKind.BUDGET,
                "无法满足跨厂商多样性约束，请调整场景模型配置",
            )

        judge_chain: list[ModelSpec] = []
        if scenario.judge:
            judge_estimate = self.estimate_cost(
                scenario.judge,
                input_tokens * 2,
                scenario.max_output_tokens,
            )
            if judge_estimate > remaining:
                raise ReviewError(
                    ErrorKind.BUDGET,
                    "预算不足以执行裁判汇总；请提高 max_cost_usd 或更换裁判模型",
                )
            judge_chain = self._chain(scenario.judge, scenario.fallbacks)

        return ReviewPlan(
            scenario=scenario,
            reviewer_chains=[
                self._chain(spec, scenario.fallbacks) for spec in selected
            ],
            judge_chain=judge_chain,
        )
