from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Constraints(BaseModel):
    max_cost_usd: float = Field(default=0.10, gt=0, le=100)
    max_latency_ms: int = Field(default=60_000, ge=1_000, le=600_000)


class Submission(BaseModel):
    content: str = Field(min_length=1, max_length=200_000)
    language: str = "zh"
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)
    callback_url: str | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content 不能为空白")
        return value.strip()


@dataclass(frozen=True)
class ModelSpec:
    model: str
    vendor: str
    role: str = "通用评审"
    input_per_million: float = 0.0
    output_per_million: float = 0.0


@dataclass
class Scenario:
    name: str
    description: str
    reviewers: list[ModelSpec]
    fallbacks: list[ModelSpec]
    judge: ModelSpec | None
    max_output_tokens: int = 1200
    reviewer_prompt: str = ""
    judge_prompt: str = ""
    similarity_threshold: float = 0.72
    min_agreement: int = 2


@dataclass
class ModelResponse:
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str = "stop"
    slot: str = ""
    role: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    requested_model: str = ""


@dataclass
class ReviewPlan:
    scenario: Scenario
    reviewer_chains: list[list[ModelSpec]]
    judge_chain: list[ModelSpec]


TaskStatus = Literal[
    "received", "routing", "reviewing", "judging", "done", "failed"
]
