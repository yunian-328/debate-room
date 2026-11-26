from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import ModelResponse, Submission


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  scenario TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  constraints_json TEXT NOT NULL,
  status TEXT NOT NULL,
  callback_url TEXT,
  idempotency_key TEXT UNIQUE,
  error TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE TABLE IF NOT EXISTS model_outputs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  slot TEXT NOT NULL,
  model TEXT NOT NULL,
  role TEXT,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  latency_ms INTEGER NOT NULL DEFAULT 0,
  output TEXT,
  finish_reason TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
  task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
  judge_model TEXT,
  score REAL,
  summary TEXT,
  verdict_json TEXT NOT NULL,
  confidence TEXT,
  needs_human INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT REFERENCES tasks(task_id),
  model TEXT,
  error_type TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  detail TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outputs_task ON model_outputs(task_id);
CREATE INDEX IF NOT EXISTS idx_outputs_model ON model_outputs(model, id DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def create_task(
        self,
        scenario: str,
        submission: Submission,
        idempotency_key: str | None = None,
    ) -> tuple[str, bool]:
        task_id = uuid.uuid4().hex
        payload = {
            "content": submission.content,
            "language": submission.language,
            "context": submission.context,
        }
        try:
            with self.connect() as db:
                db.execute(
                    """INSERT INTO tasks
                    (task_id, scenario, payload_json, constraints_json, status,
                     callback_url, idempotency_key, created_at)
                    VALUES (?, ?, ?, ?, 'received', ?, ?, ?)""",
                    (
                        task_id,
                        scenario,
                        self._json(payload),
                        submission.constraints.model_dump_json(),
                        submission.callback_url,
                        idempotency_key,
                        utc_now(),
                    ),
                )
            return task_id, True
        except sqlite3.IntegrityError:
            if not idempotency_key:
                raise
            with self.connect() as db:
                row = db.execute(
                    "SELECT * FROM tasks WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
            if not row:
                raise
            if (row["scenario"] != scenario or json.loads(row["payload_json"]) != payload
                or json.loads(row["constraints_json"]) != submission.constraints.model_dump()
                or row["callback_url"] != submission.callback_url):
                raise ValueError("幂等键已用于不同的请求")
            return row["task_id"], False

    def _decode_task(self, row: sqlite3.Row) -> dict[str, Any]:
        task = dict(row)
        task["payload"] = json.loads(task.pop("payload_json"))
        task["constraints"] = json.loads(task.pop("constraints_json"))
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._decode_task(row) if row else None

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_task(row) for row in rows]

    def unfinished_task_ids(self) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT task_id FROM tasks WHERE status NOT IN ('done', 'failed')"
            ).fetchall()
        return [row["task_id"] for row in rows]

    def update_status(self, task_id: str, status: str, error: str | None = None) -> None:
        finished_at = utc_now() if status in {"done", "failed"} else None
        with self.connect() as db:
            db.execute(
                """UPDATE tasks SET status=?, error=?,
                finished_at=COALESCE(?, finished_at) WHERE task_id=?""",
                (status, error, finished_at, task_id),
            )

    def record_output(
        self,
        task_id: str,
        response: ModelResponse,
        *,
        status: str = "ok",
        output: str | None = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO model_outputs
                (task_id, slot, model, role, prompt_tokens, completion_tokens,
                 cost_usd, latency_ms, output, finish_reason, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    response.slot,
                    response.model,
                    response.role,
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.cost_usd,
                    response.latency_ms,
                    response.content if output is None else output,
                    response.finish_reason,
                    status,
                    utc_now(),
                ),
            )

    def record_error(
        self,
        task_id: str,
        model: str,
        error_type: str,
        attempt: int,
        detail: str,
    ) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO errors
                (task_id, model, error_type, attempt, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, model, error_type, attempt, detail[:4000], utc_now()),
            )

    def save_verdict(self, task_id: str, verdict: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO verdicts
                (task_id, judge_model, score, summary, verdict_json,
                 confidence, needs_human, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  judge_model=excluded.judge_model, score=excluded.score,
                  summary=excluded.summary, verdict_json=excluded.verdict_json,
                  confidence=excluded.confidence, needs_human=excluded.needs_human,
                  created_at=excluded.created_at""",
                (
                    task_id,
                    verdict.get("judge_model", "cluster"),
                    verdict.get("score"),
                    verdict.get("summary", ""),
                    self._json(verdict),
                    verdict.get("confidence", "low"),
                    int(bool(verdict.get("needs_human"))),
                    utc_now(),
                ),
            )

    def get_task_detail(self, task_id: str) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if not task:
            return None
        with self.connect() as db:
            outputs = db.execute(
                "SELECT * FROM model_outputs WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
            errors = db.execute(
                "SELECT * FROM errors WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
            verdict = db.execute(
                "SELECT * FROM verdicts WHERE task_id=?", (task_id,)
            ).fetchone()
        task["model_outputs"] = [dict(row) for row in outputs]
        task["errors"] = [dict(row) for row in errors]
        task["total_cost_usd"] = round(
            sum(row["cost_usd"] for row in task["model_outputs"]), 8
        )
        if verdict:
            task["verdict"] = json.loads(verdict["verdict_json"])
        else:
            task["verdict"] = None
        return task

    def model_health(self, model: str, window: int = 20) -> dict[str, float | int]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT status, latency_ms FROM model_outputs
                WHERE model=? ORDER BY id DESC LIMIT ?""",
                (model, window),
            ).fetchall()
        attempts = len(rows)
        failures = sum(row["status"] != "ok" for row in rows)
        return {
            "attempts": attempts,
            "error_rate": failures / attempts if attempts else 0.0,
            "avg_latency_ms": (
                sum(row["latency_ms"] for row in rows) / attempts if attempts else 0.0
            ),
        }

    def stats(self) -> dict[str, Any]:
        with self.connect() as db:
            by_status = db.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
            totals = db.execute(
                """SELECT COUNT(*) AS calls,
                COALESCE(SUM(cost_usd), 0) AS cost,
                COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens
                FROM model_outputs WHERE status='ok'"""
            ).fetchone()
            by_model = db.execute(
                """SELECT model, COUNT(*) AS calls, ROUND(SUM(cost_usd), 8) AS cost_usd,
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms
                FROM model_outputs WHERE status='ok' GROUP BY model ORDER BY calls DESC"""
            ).fetchall()
        return {
            "tasks": {row["status"]: row["count"] for row in by_status},
            "successful_model_calls": totals["calls"],
            "total_cost_usd": round(totals["cost"], 8),
            "total_tokens": totals["tokens"],
            "models": [dict(row) for row in by_model],
        }
