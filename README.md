# Debate Room · 多模型内容质检工作台

通过**多个相互关联的 AI 场景**对提交内容进行评审：文案合规评审、事实核查、代码变更评审。
项目按要求包含六个部分：**自定义网络钩子、模型路由器、数据存储、错误处理程序、OpenRouter（及自定义端点）调用层、三个相互关联的场景**。

## 六个组成部分

1. **自定义网络钩子**：`POST /api/v1/hooks/form`（文案评审）、`/api/v1/hooks/fact-check`（事实核查）、`/api/v1/hooks/code-review`（代码评审），支持幂等键 `Idempotency-Key`。
2. **模型路由器**（`app/router.py`）：按健康度、预算与跨厂商多样性约束选择评审模型与裁判。
3. **数据存储**（`app/storage.py`）：SQLite 持久化任务、模型调用、成本统计。
4. **错误处理程序**（`app/errors.py`）：错误分类（超时/限流/鉴权/供应商故障/内容过滤）与分级重试。
5. **OpenRouter 调用层**（`app/openrouter.py`）：支持 OpenRouter 官方端点，也支持任意 OpenAI 兼容自定义端点（如 DeepSeek：设置 `OPENROUTER_BASE_URL=https://api.deepseek.com` 与 `CUSTOM_MODELS=deepseek-chat`）。内置确定性 Mock 客户端用于离线演示。
6. **三个相互关联的场景**（`config/scenarios.yaml`）：`copy_review`、`fact_check`、`code_review`，共享同一套钩子/路由/存储/错误处理管线。

## 快速开始（本地）

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Mock 模式（免费、离线、确定性）：直接运行
copy .env.example .env   # 保持 OPENROUTER_MOCK=true
.venv\Scripts\python.exe -m uvicorn app.main:app --port 7100
```

打开 http://127.0.0.1:7100/workbench 使用网页工作台，或 http://127.0.0.1:7100/docs 查看 API。

## 真实模型模式

在 `.env` 中设置（**不要提交真实 .env**）：

```
OPENROUTER_MOCK=false
OPENROUTER_API_KEY=<你的 OpenRouter 或 DeepSeek Key>
DEBATE_ROOM_API_TOKEN=<32 位以上随机令牌>
# DeepSeek 直连（无需 OpenRouter）：
OPENROUTER_BASE_URL=https://api.deepseek.com
CUSTOM_MODELS=deepseek-chat
FACT_SEARCH_PROVIDER=none
```

## 部署（Docker / Render）

仓库根目录自带 `Dockerfile`：

```bash
docker build -t debate-room .
docker run -p 8000:8000 --env-file .env debate-room
```

Render 等平台可直接关联本仓库自动构建，环境变量在平台后台配置。

## 测试

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

## 演示脚本

```bash
.venv\Scripts\python.exe demo.py
```
