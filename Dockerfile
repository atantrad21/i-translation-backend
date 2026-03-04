# I-Translation Backend - Medical Image Converter v6.0
# FORCE REBUILD - Complete fresh build
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start application - PORT variable expansion via shell
CMD sh -c "echo 'Starting on PORT:' $PORT && gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 3600 --workers 1 --threads 4 --worker-class gthread app:app"
