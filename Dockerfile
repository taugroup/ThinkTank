# Policy Think Tank — local-first Streamlit app.
# Ollama is expected to run on the host (or a sibling container); set OLLAMA_HOST.
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; pandoc only needed for DOCX export.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Local-first defaults; no secrets baked in.
ENV POLICY_MOCK_MODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
