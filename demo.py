from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from app.config import PROJECT_ROOT
from app.models import Submission
from app.openrouter import MockOpenRouterClient
from app.pipeline import ReviewPipeline
from app.router import ModelRouter
from app.scenarios import load_scenarios
from app.storage import Storage


EXAMPLES = {
    "copy_review": "全新一代智能水杯，让每一口水都更健康。立即下载 App，开启科学饮水生活。",
    "code_review": """diff --git a/app/login.py b/app/login.py
@@ -1,3 +1,5 @@
 def login(request):
-    return authenticate(request.user)
+    username = request.data["username"]
+    password = request.data["password"]
+    return authenticate(username, password)
""",
    "fact_check": "每天喝八杯水对所有人的健康都是必需的。",
}


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="debate-room-demo-") as directory:
        storage = Storage(Path(directory) / "demo.db")
        scenarios = load_scenarios(PROJECT_ROOT / "config" / "scenarios.yaml")
        pipeline = ReviewPipeline(
            storage,
            ModelRouter(storage),
            MockOpenRouterClient(),
            scenarios,
            max_retries_per_model=0,
        )
        for scenario, content in EXAMPLES.items():
            task_id, _ = storage.create_task(scenario, Submission(content=content))
            await pipeline.process(task_id)
            detail = storage.get_task_detail(task_id)
            verdict = detail["verdict"] or {}
            print("\n" + "=" * 68)
            print(f"场景：{scenario} ｜ 状态：{detail['status']} ｜ 任务：{task_id}")
            print(f"模型调用：{len(detail['model_outputs'])} 次 ｜ 模拟成本：${detail['total_cost_usd']:.6f}")
            print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
