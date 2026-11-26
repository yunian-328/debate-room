from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .models import ModelSpec, Scenario


def _model(data: dict[str, Any] | None) -> ModelSpec | None:
    if not data:
        return None
    return ModelSpec(
        model=data["model"],
        vendor=data.get("vendor") or data["model"].split("/", 1)[0],
        role=data.get("role", "通用评审"),
        input_per_million=float(data.get("input_per_million", 0)),
        output_per_million=float(data.get("output_per_million", 0)),
    )


def load_scenarios(path: Path) -> dict[str, Scenario]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    scenarios: dict[str, Scenario] = {}
    for name, data in raw["scenarios"].items():
        scenarios[name] = Scenario(
            name=name,
            description=data.get("description", ""),
            reviewers=[_model(item) for item in data.get("reviewers", [])],
            fallbacks=[_model(item) for item in data.get("fallbacks", [])],
            judge=_model(data.get("judge")),
            max_output_tokens=int(data.get("max_output_tokens", 1200)),
            reviewer_prompt=data.get("reviewer_prompt", ""),
            judge_prompt=data.get("judge_prompt", ""),
            similarity_threshold=float(data.get("similarity_threshold", 0.72)),
            min_agreement=int(data.get("min_agreement", 2)),
        )
    return scenarios

FREE_PREVIEW_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "baidu/ernie-5.5-flash-preview:free",
    "particle-ai/particle-1:free",
)


def use_free_preview_models(scenarios: dict[str, Scenario]) -> dict[str, Scenario]:
    def replace_model(spec: ModelSpec, model: str) -> ModelSpec:
        return ModelSpec(model=model, vendor=model.split("/", 1)[0], role=spec.role)

    return {
        name: replace(
            scenario,
            reviewers=[replace_model(spec, model) for spec, model in zip(scenario.reviewers, FREE_PREVIEW_MODELS)],
            fallbacks=[],
            judge=replace_model(scenario.judge, FREE_PREVIEW_MODELS[0]) if scenario.judge else None,
        )
        for name, scenario in scenarios.items()
    }


# DeepSeek 官方价格（USD / 百万 token，非缓存命中）
CUSTOM_INPUT_PER_MILLION = 0.27
CUSTOM_OUTPUT_PER_MILLION = 1.10


def use_custom_models(scenarios: dict[str, Scenario], models: list[str]) -> dict[str, Scenario]:
    """将自定义端点（如 DeepSeek 原生 API）的模型注入所有场景。

    自定义端点通常只有一个可用模型，通过给每个槽位分配独立的
    vendor 标签（custom-1/2/3）满足路由器的跨厂商多样性约束。
    """
    pool = [m for m in models if m] or ["deepseek-chat"]

    def replace_model(spec: ModelSpec, index: int) -> ModelSpec:
        return ModelSpec(
            model=pool[index % len(pool)],
            vendor=f"custom-{index + 1}",
            role=spec.role,
            input_per_million=CUSTOM_INPUT_PER_MILLION,
            output_per_million=CUSTOM_OUTPUT_PER_MILLION,
        )

    return {
        name: replace(
            scenario,
            reviewers=[replace_model(spec, i) for i, spec in enumerate(scenario.reviewers)],
            fallbacks=[],
            judge=replace_model(scenario.judge, 0) if scenario.judge else None,
        )
        for name, scenario in scenarios.items()
    }
