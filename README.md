# I-Translation Backend

Medical image translation backend using TensorFlow GANs (Checkpoint 652).

## Features
- CT → MRI conversion
- MRI → CT conversion
- 4 independent U-Net generators (F, G, I, J)
- Checkpoint 652 trained weights

## Deployment

### Render.com
1. Push to GitHub
2. Connect repository to Render
3. Select "Web Service"
4. Build command: `pip install -r requirements.txt`
5. Start command: Uses `Procfile`

### Environment Variables
- `PORT` - Auto-set by Render

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "online",
  "models_loaded": true,
  "loaded_generators": ["F", "G", "I", "J"],
  "version": "4.7.7-ROBUST"
}
```

### Convert Images
```bash
POST /convert
Content-Type: multipart/form-data

Fields:
- image1: File (PNG/JPG/DICOM)
- image2: File (optional)
- image3: File (optional)
- image4: File (optional)
- type: "ct_to_mri" or "mri_to_ct"
```

Response:
```json
{
  "image1": {
    "F": "<hex_encoded_png>",
    "G": "<hex_encoded_png>",
    "I": "<hex_encoded_png>",
    "J": "<hex_encoded_png>"
  }
}
```

## Model Architecture
- Input: 256×256 RGB
- Output: 256×256 RGB
- Architecture: U-Net with Instance Normalization
- Training checkpoint: 652

## Files
- `app.py` - Flask application
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version
- `Procfile` - Gunicorn configuration

## Contact
atantrad@gmail.com
