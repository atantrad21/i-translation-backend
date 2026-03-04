# Railway Dockerfile - Complete control over build and runtime
FROM python:3.9-slim

WORKDIR /app

# Install bash (slim images don't have it)
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py .
COPY requirements.txt .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Expose port (Railway will set $PORT dynamically)
EXPOSE 8080

# Run our custom start script
CMD ["bash", "start.sh"]
