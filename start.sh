#!/bin/bash
set -e

echo "Creating virtual environment..."
python3.9 -m venv /opt/venv

echo "Activating virtual environment..."
source /opt/venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Starting application..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 3600 --workers 1 --threads 4 --worker-class gthread app:app
