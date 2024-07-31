FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app:api.app", "--host", "0.0.0.0", "--port", "8000"]


