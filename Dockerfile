# Must be Python 3.10 for TensorFlow 2.10 compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV/Pillow and downloading tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Flask/Gunicorn will run on
EXPOSE 7860

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
