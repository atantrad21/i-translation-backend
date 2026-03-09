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

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("🚀 I-TRANSLATION BACKEND v4.8.9 (LAZY LOADING - FIXED)")
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
LOADING_IN_PROGRESS = False
LOADING_LOCK = threading.Lock()

def download_from_google_drive(file_id, output_path):
    logger.info(f"📥 Starting download for file_id: {file_id}")
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    session = requests.Session()
    response = session.get(url, stream=True, timeout=120)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = url + '&confirm=' + value
            response = session.get(url, stream=True, timeout=120)
            break
    chunk_size = 32768
    total_size = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
                if total_size % (10 * 1024 * 1024) == 0:
                    logger.info(f"   Downloaded: {total_size / (1024*1024):.1f} MB")
    logger.info(f"✅ Download complete: {total_size / (1024*1024):.2f} MB")
    return total_size > 0

def initialize_models():
    global MODELS, MODELS_LOADED, LOADING_IN_PROGRESS
    
    with LOADING_LOCK:
        if MODELS_LOADED:
            logger.info("✅ Models already loaded")
            return True
        
        if LOADING_IN_PROGRESS:
            logger.info("⏳ Loading already in progress...")
            return False
        
        LOADING_IN_PROGRESS = True
    
    try:
        logger.info("=" * 70)
        logger.info("🔧 INITIALIZING MODELS (LAZY LOADING)")
        logger.info("=" * 70)
        
        logger.info("📦 Creating temp directory for weights...")
        temp_dir = '/tmp/weights'
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"✅ Temp directory created: {temp_dir}")
        
        weight_paths = {}
        for name, file_id in WEIGHT_FILES.items():
            output_path = os.path.join(temp_dir, 'generator_' + name.lower() + '.h5')
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Generator {name} already exists: {size_mb:.2f} MB")
                weight_paths[name] = output_path
                continue
            
            logger.info(f"📥 Downloading Generator {name}...")
            logger.info(f"   File ID: {file_id}")
            logger.info(f"   Output: {output_path}")
            
            success = download_from_google_drive(file_id, output_path)
            
            if success and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Generator {name} downloaded: {size_mb:.2f} MB")
                weight_paths[name] = output_path
            else:
                logger.error(f"❌ Generator {name} download FAILED")
                LOADING_IN_PROGRESS = False
                return False
        
        logger.info("=" * 70)
        logger.info("🏗️  BUILDING AND LOADING GENERATORS")
        logger.info("=" * 70)
        
        for name in ['F', 'G', 'I', 'J']:
            logger.info(f"🔨 Building Generator {name} architecture...")
            generator = unet_generator()
            logger.info(f"✅ Generator {name} architecture built")
            
            logger.info(f"📂 Loading weights from: {weight_paths[name]}")
            generator.load_weights(weight_paths[name])
            logger.info(f"✅ Generator {name} weights loaded")
            
            MODELS[name] = generator
            logger.info(f"✅ Generator {name} SUCCESS")
        
        MODELS_LOADED = True
        LOADING_IN_PROGRESS = False
        logger.info("=" * 70)
        logger.info(f"🎉 ALL MODELS LOADED: {len(MODELS)}/4")
        logger.info("=" * 70)
        return True
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ INITIALIZATION FAILED")
        logger.error(f"❌ Error: {str(e)}")
        logger.error("=" * 70)
        import traceback
        traceback.print_exc()
        MODELS_LOADED = False
        LOADING_IN_PROGRESS = False
        return False

logger.info("✅ Flask app initialization starting (models will load on first request)")

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
        'loading_in_progress': LOADING_IN_PROGRESS,
        'generators': {
            'F': 'F' in MODELS,
            'G': 'G' in MODELS,
            'I': 'I' in MODELS,
            'J': 'J' in MODELS
        },
        'version': 'v4.8.9-lazy'
    })

@app.route('/convert', methods=['POST'])
def convert():
    global MODELS_LOADED
    
    if not MODELS_LOADED and not LOADING_IN_PROGRESS:
        logger.info("🔄 First request received - starting model initialization...")
        success = initialize_models()
        if not success:
            return jsonify({'error': 'Model initialization failed', 'retry': True}), 503
    
    if LOADING_IN_PROGRESS:
        return jsonify({'error': 'Models are loading, please wait...', 'retry': True}), 503
    
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded yet', 'retry': True}), 503
    
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
        logger.error('Conversion error: ' + str(e))
        return jsonify({'error': str(e)}), 500

logger.info("✅ Flask app created - ready to accept requests")
logger.info("📌 Models will load on first /convert request")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
