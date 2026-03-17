# Upgrade to Python 3.10 to support modern TensorFlow 2.16+ (Keras 3)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV/Pillow and gdown
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
