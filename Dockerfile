# I-Translation v7.0 - CACHE BUSTER
# Using specific Python version to force fresh build
FROM python:3.9.20-slim

WORKDIR /app

# Copy files
COPY app.py .
COPY requirements.txt .

# Install with no cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Start with PORT expansion
CMD ["sh", "-c", "echo 'v7.0 Starting on PORT:' $PORT && gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 3600 --workers 1 --threads 4 --worker-class gthread app:app"]
