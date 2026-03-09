"""
I-Translation Backend v4.8.4 - STARTUP MODEL LOADING
=====================================================
FIX: Models initialize at STARTUP (before Flask app) instead of lazy loading
- Solves stuck issue where models never loaded after service went live
- All 4 generators download and load BEFORE server accepts requests
- Users Google Drive file IDs embedded (publicly accessible)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os
import gdown
import logging

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM LAYERS - Instance Normalization
# ============================================================================
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

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

# ============================================================================
# MODEL ARCHITECTURE - U-Net Generator (64x64 Grayscale)
# ============================================================================
def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def unet_generator():
    inputs = layers.Input(shape=[64, 64, 1])
    
    # Downsampling: 6 layers
    down_stack = [
        downsample(128, 4, apply_norm=False),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]
    
    # Upsampling: 5 layers
    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'
    )
    
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

# ============================================================================
# GOOGLE DRIVE FILE IDs - USER PUBLIC FILES
# ============================================================================
WEIGHT_FILES = {
    'F': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'G': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'I': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'J': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j',
}

# ============================================================================
# GLOBAL MODEL STORAGE
# ============================================================================
MODELS = {}
MODELS_LOADED = False

# ============================================================================
# MODEL INITIALIZATION - Downloads and loads all 4 generators
# ============================================================================
def initialize_models():
    global MODELS, MODELS_LOADED
    
    logger.info("=" * 60)
    logger.info("INITIALIZING MODELS AT STARTUP...")
    logger.info("=" * 60)
    
    try:
        # STEP 1: Download weights from Google Drive
        logger.info("[DOWNLOAD] Starting weight downloads from Google Drive...")
        temp_dir = '/tmp/weights'
        os.makedirs(temp_dir, exist_ok=True)
        
        weight_paths = {}
        for name, file_id in WEIGHT_FILES.items():
            output_path = os.path.join(temp_dir, f'generator_{name.lower()}.h5')
            url = f'https://drive.google.com/uc?id={file_id}'
            
            logger.info(f"[{name}] Downloading from {file_id}...")
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"[{name}] Downloaded successfully ({size_mb:.2f} MB)")
                weight_paths[name] = output_path
            else:
                logger.error(f"[{name}] Download failed!")
                return False
        
        # STEP 2: Build and load generators
        logger.info("[MODELS] Building and loading generators...")
        for name in ['F', 'G', 'I', 'J']:
            logger.info(f"[{name}] Building U-Net architecture...")
            generator = unet_generator()
            
            logger.info(f"[{name}] Loading weights from {weight_paths[name]}...")
            generator.load_weights(weight_paths[name])
            
            MODELS[name] = generator
            logger.info(f"[{name}] SUCCESS!")
        
        MODELS_LOADED = True
        logger.info("=" * 60)
        logger.info(f"MODELS LOADED: {len(MODELS)}/4")
        logger.info("[STARTUP] ALL SYSTEMS READY!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"[STARTUP] INITIALIZATION FAILED: {str(e)}")
        MODELS_LOADED = False
        return False

# ============================================================================
# INITIALIZE MODELS AT STARTUP (BEFORE FLASK APP)
# ============================================================================
logger.info("=" * 60)
logger.info("STARTING I-TRANSLATION BACKEND v4.8.4")
logger.info("=" * 60)
initialize_models()

# ============================================================================
# FLASK APP CREATION (AFTER MODELS ARE LOADED)
# ============================================================================
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)

def postprocess_image(tensor):
    tensor = (tensor<sup>0</sup> + 1.0) * 127.5
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    img = Image.fromarray(tensor[:, :, 0], mode='L')
    img = img.resize((256, 256), Image.LANCZOS)
    
    output = io.BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'models_loaded': MODELS_LOADED,
        'generators': {
            'F': 'F' in MODELS,
            'G': 'G' in MODELS,
            'I': 'I' in MODELS,
            'J': 'J' in MODELS,
        },
        'version': 'v4.8.4',
        'architecture': '64x64 grayscale 6 down 5 up layers'
    })

@app.route('/convert', methods=['POST'])
def convert():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded yet'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        input_tensor = preprocess_image(image_bytes)
        
        results = {}
        for name in ['F', 'G', 'I', 'J']:
            output_tensor = MODELS[name](input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor.numpy())
            results[name] = output_bytes.hex()
        
        return jsonify({
            'success': True,
            'outputs': results
        })
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SERVER STARTUP
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
