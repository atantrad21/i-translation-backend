"""
================================================================================
I-TRANSLATION v6.1 - LOAD COMPLETE MODELS (NOT JUST WEIGHTS)
================================================================================
Key change: Load the .h5 files as complete models instead of just weights
This matches how the files were saved in Colab
================================================================================
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import zipfile
import os
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

GENERATORS = {}

GDRIVE_FILE_IDS = {
    'f': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'g': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
}

def download_from_gdrive_requests(file_id, output_path):
    try:
        logger.info(f"Downloading file ID: {file_id}")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                response = session.get(url, stream=True)
                break
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:
                            logger.info(f"Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Downloaded successfully! Size: {file_size:.2f} MB")
            return True
        else:
            logger.error(f"Download failed - file not created")
            return False
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return False

logger.info("="*80)
logger.info("I-TRANSLATION v6.1 - LOAD COMPLETE MODELS")
logger.info("="*80)
logger.info("Starting model download and loading process...")
logger.info(f"Total models to load: {len(GDRIVE_FILE_IDS)}")

for gen_name, file_id in GDRIVE_FILE_IDS.items():
    logger.info("="*60)
    logger.info(f"[{gen_name.upper()}] Processing Generator {gen_name.upper()}")
    logger.info("="*60)
    output_path = f'/tmp/generator_{gen_name}.h5'
    if download_from_gdrive_requests(file_id, output_path):
        try:
            logger.info(f"[{gen_name.upper()}] Loading as complete model...")
            model = tf.keras.models.load_model(
                output_path,
                custom_objects={'InstanceNormalization': InstanceNormalization},
                compile=False
            )
            logger.info(f"[{gen_name.upper()}] ✅ Model loaded successfully!")
            GENERATORS[gen_name] = model
            os.remove(output_path)
            logger.info(f"[{gen_name.upper()}] Cleaned up temporary file")
        except Exception as e:
            logger.error(f"[{gen_name.upper()}] ❌ Error loading model: {str(e)}")
    else:
        logger.error(f"[{gen_name.upper()}] Download failed, skipping model load")
    logger.info("")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(tensor):
    img_array = tensor[0]
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    img_array = img_array[:, :, 0]
    img = Image.fromarray(img_array, mode='L')
    return img

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(GENERATORS) == 4,
        'loaded_models': list(GENERATORS.keys()),
        'version': '6.1'
    })

@app.route('/convert', methods=['POST'])
def convert():
    try:
        if len(GENERATORS) != 4:
            return jsonify({'error': 'Models not fully loaded yet'}), 503
        
        conversion_type = request.form.get('type', 'ct_to_mri')
        results = {}
        
        for i in range(1, 5):
            file_key = f'image{i}'
            if file_key not in request.files:
                continue
            file = request.files[file_key]
            if file.filename == '':
                continue
            
            image_bytes = file.read()
            input_tensor = preprocess_image(image_bytes)
            
            if conversion_type == 'ct_to_mri':
                output_f = GENERATORS['f'](input_tensor, training=False)
                output_g = GENERATORS['g'](input_tensor, training=False)
                output_i = GENERATORS['i'](output_f, training=False)
                output_j = GENERATORS['j'](output_g, training=False)
            else:
                output_i = GENERATORS['i'](input_tensor, training=False)
                output_j = GENERATORS['j'](input_tensor, training=False)
                output_f = GENERATORS['f'](output_i, training=False)
                output_g = GENERATORS['g'](output_j, training=False)
            
            img_f = postprocess_image(output_f.numpy())
            img_g = postprocess_image(output_g.numpy())
            img_i = postprocess_image(output_i.numpy())
            img_j = postprocess_image(output_j.numpy())
            
            results[file_key] = {
                'generator_f': img_f,
                'generator_g': img_g,
                'generator_i': img_i,
                'generator_j': img_j
            }
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for img_key, generators in results.items():
                for gen_name, img in generators.items():
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    zip_file.writestr(f'{img_key}_{gen_name}.png', img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='converted_images.zip')
    
    except Exception as e:
        logger.error(f"Error in convert: {str(e)}")
        return jsonify({'error': str(e)}), 500

logger.info("="*80)
logger.info("✅ APPLICATION READY TO SERVE REQUESTS")
logger.info("="*80)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
