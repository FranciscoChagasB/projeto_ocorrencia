FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch leve (CPU) para manter a imagem pequena
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Expõe a porta interna fixa
EXPOSE 8080

# Força o Uvicorn a rodar na porta 8080 internamente
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]