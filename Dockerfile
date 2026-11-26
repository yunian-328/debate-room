FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home debate \
    && mkdir -p /app/data \
    && chown debate:debate /app/data
COPY app ./app
COPY config ./config
COPY smoke_openrouter.py .
USER debate
EXPOSE 8000
# Shell 形式展开 $PORT：云平台（Render 等）注入 PORT 时自动使用，本地回退 8000
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
