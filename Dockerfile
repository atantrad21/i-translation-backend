# Railway Dockerfile - Download models at runtime
FROM python:3.9

WORKDIR /app

# Copy application files
COPY app.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start gunicorn - models will download on first startup via app.py
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 3600 --workers 1 --threads 4 --worker-class gthread app:app
