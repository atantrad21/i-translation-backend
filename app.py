from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os
import requests
import logging
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("🚀 I-TRANSLATION BACKEND v4.9.1 (FIXED DOWNLOADS)")
print("=" * 70)

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='ones', trainable=True)
        self.offset = self.add_weight(name='offset', shape=(input_shape[-1],), initializer='zeros', trainable=True)
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset
    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def unet_generator():
    inputs = layers.Input(shape=[64, 64, 1])
    down_stack = [downsample(128, 4, apply_norm=False), downsample(256, 4), downsample(256, 4), downsample(256, 4), downsample(256, 4), downsample(256, 4)]
    up_stack = [upsample(256, 4, apply_dropout=True), upsample(256, 4, apply_dropout=True), upsample(256, 4, apply_dropout=True), upsample(256, 4), upsample(128, 4)]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)

WEIGHT_FILES = {
    'F': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'G': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'I': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'J': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
}

MODELS = {}
MODELS_LOADED = False
LOADING_ERROR = None
LOADING_PROGRESS = "Starting..."

def download_from_google_drive_fixed(file_id, output_path, max_retries=3):
    """Download from Google Drive with proper error handling and retries."""
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Attempt {attempt + 1}/{max_retries} - Downloading file_id: {file_id}")
            
            # Try direct download first
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            session = requests.Session()
            
            # Initial request
            response = session.get(url, stream=True, timeout=60)
            
            # Check if we need confirmation token
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            # If token found, make confirmed request
            if token:
                logger.info(f"   Using confirmation token: {token[:20]}...")
                url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm={token}'
                response = session.get(url, stream=True, timeout=60)
            
            # Check response status
            if response.status_code != 200:
                logger.warning(f"   HTTP {response.status_code}, retrying...")
                time.sleep(5)
                continue
            
            # Download file
            chunk_size = 32768
            total_size = 0
            logger.info(f"   Downloading to: {output_path}")
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        if total_size % (10 * 1024 * 1024) == 0:
                            logger.info(f"   Progress: {total_size / (1024*1024):.1f} MB")
            
            # Verify download
            if total_size < 1000000:  # Less than 1 MB
                logger.warning(f"   File too small ({total_size} bytes), likely HTML error page")
                if os.path.exists(output_path):
                    with open(output_path, 'r', errors='ignore') as f:
                        content_preview = f.read(500)
                        if '<html' in content_preview.lower():
                            logger.error(f"   Downloaded HTML instead of model file!")
                time.sleep(5)
                continue
            
            logger.info(f"✅ Download complete: {total_size / (1024*1024):.2f} MB")
            return True
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ Attempt {attempt + 1} timeout after 60 seconds")
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            wait_time = 10 * (attempt + 1)
            logger.info(f"   Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    logger.error(f"❌ All {max_retries} download attempts failed for file_id: {file_id}")
    return False

def initialize_models_background():
    global MODELS, MODELS_LOADED, LOADING_ERROR, LOADING_PROGRESS
    
    try:
        LOADING_PROGRESS = "Creating temp directory..."
        logger.info("=" * 70)
        logger.info("🔧 INITIALIZING MODELS (BACKGROUND THREAD)")
        logger.info("=" * 70)
        
        temp_dir = '/tmp/weights'
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"✅ Temp directory created: {temp_dir}")
        
        weight_paths = {}
        for name, file_id in WEIGHT_FILES.items():
            LOADING_PROGRESS = f"Downloading Generator {name}..."
            output_path = os.path.join(temp_dir, f'generator_{name.lower()}.h5')
            
            # Check if already downloaded and valid
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if size_mb > 10:  # At least 10 MB
                    logger.info(f"✅ Generator {name} cached: {size_mb:.2f} MB")
                    weight_paths[name] = output_path
                    continue
                else:
                    logger.warning(f"⚠️  Cached file too small ({size_mb:.2f} MB), re-downloading...")
                    os.remove(output_path)
            
            logger.info(f"\n📥 Downloading Generator {name}...")
            logger.info(f"   File ID: {file_id}")
            
            success = download_from_google_drive_fixed(file_id, output_path)
            
            if success and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Generator {name} downloaded: {size_mb:.2f} MB")
                weight_paths[name] = output_path
            else:
                error_msg = f"Generator {name} download failed after all retries"
                logger.error(f"❌ {error_msg}")
                LOADING_ERROR = error_msg
                LOADING_PROGRESS = f"Failed: {error_msg}"
                return False
        
        LOADING_PROGRESS = "Building generators..."
        logger.info("\n" + "=" * 70)
        logger.info("🏗️  BUILDING AND LOADING GENERATORS")
        logger.info("=" * 70)
        
        for name in ['F', 'G', 'I', 'J']:
            LOADING_PROGRESS = f"Building Generator {name}..."
            logger.info(f"\n🔨 Building Generator {name} architecture...")
            generator = unet_generator()
            logger.info(f"✅ Generator {name} architecture built")
            
            LOADING_PROGRESS = f"Loading weights for Generator {name}..."
            logger.info(f"📂 Loading weights from: {weight_paths[name]}")
            generator.load_weights(weight_paths[name])
            logger.info(f"✅ Generator {name} weights loaded")
            
            MODELS[name] = generator
            logger.info(f"✅ Generator {name} SUCCESS")
        
        MODELS_LOADED = True
        LOADING_PROGRESS = "Complete!"
        logger.info("\n" + "=" * 70)
        logger.info(f"🎉 ALL MODELS LOADED: {len(MODELS)}/4")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 70)
        import traceback
        traceback.print_exc()
        LOADING_ERROR = error_msg
        LOADING_PROGRESS = f"Failed: {error_msg}"
        MODELS_LOADED = False
        return False

# Start background loading immediately
logger.info("🔄 Starting background model loading thread...")
loading_thread = threading.Thread(target=initialize_models_background, daemon=True)
loading_thread.start()
logger.info("✅ Background thread started")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)

def postprocess_image(tensor):
    img_data = tensor.numpy()
    img_data = np.squeeze(img_data, axis=0)
    img_data = (img_data + 1.0) * 127.5
    img_data = np.clip(img_data, 0, 255)
    img_data = img_data.astype(np.uint8)
    img_data = np.squeeze(img_data, axis=-1)
    img = Image.fromarray(img_data, mode='L')
    img = img.resize((256, 256), Image.LANCZOS)
    output = io.BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'models_loaded': MODELS_LOADED,
        'loading_progress': LOADING_PROGRESS,
        'loading_error': LOADING_ERROR,
        'generators': {
            'F': 'F' in MODELS,
            'G': 'G' in MODELS,
            'I': 'I' in MODELS,
            'J': 'J' in MODELS
        },
        'version': 'v4.9.1-fixed'
    })

@app.route('/convert', methods=['POST'])
def convert():
    if LOADING_ERROR:
        return jsonify({'error': f'Model loading failed: {LOADING_ERROR}', 'retry': False}), 500
    
    if not MODELS_LOADED:
        return jsonify({
            'error': 'Models are still loading, please wait...',
            'progress': LOADING_PROGRESS,
            'retry': True
        }), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
        results = {}
        for name in ['F', 'G', 'I', 'J']:
            output_tensor = MODELS[name](input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor)
            results[name] = output_bytes.hex()
        return jsonify({'success': True, 'outputs': results})
    except Exception as e:
        logger.error(f'Conversion error: {str(e)}')
        return jsonify({'error': str(e)}), 500

logger.info("✅ Flask app created - models loading in background")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
