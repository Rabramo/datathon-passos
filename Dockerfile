FROM python:3.12-slim

WORKDIR /app

# Dependências mínimas (pyarrow pode precisar de wheels; build-essential ajuda)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]