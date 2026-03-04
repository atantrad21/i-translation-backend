# Railway Dockerfile - Build models during build phase
FROM python:3.9

WORKDIR /app

# Copy application files
COPY app.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download all 4 models during BUILD phase (not runtime)
RUN python3 << 'EOF'
import gdown
import os

FILE_IDS = {
    'f': '1dMvJtRBb32BnGI8xc5lJd0U-NbJh90fT',
    'g': '11VoWUJ5Iq30HgBfLyTF5mnczk7DLiOFN',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1tNSVLfubqvFv5ACR_8B8Dp47UnsC9-He'
}

print("📥 Downloading 4 generator models (800 checkpoints each)...")
for name, file_id in FILE_IDS.items():
    output_path = f'/tmp/generator_{name}_800ckpt.h5'
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f'Downloading generator_{name}...')
    gdown.download(url, output_path, quiet=False)
    print(f'✅ generator_{name} downloaded ({os.path.getsize(output_path) / (1024*1024):.2f} MB)')

print("✅ All 4 models downloaded successfully!")
EOF

# Expose port
EXPOSE 8080

# Start gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 3600 --workers 1 --threads 4 --worker-class gthread app:app
