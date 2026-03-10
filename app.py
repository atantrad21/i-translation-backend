from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os
import logging
import gdown

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("🚀 I-TRANSLATION BACKEND v6.0 - CHECKPOINT 652")
print("=" * 70)

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

def downsample(filters, size, apply_norm=True):
    result = keras.Sequential()
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def unet_generator():
    inputs = layers.Input(shape=[64, 64, 1])
    
    down_stack = [
        downsample(128, 4, apply_norm=False),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]
    
    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(256, 4),
        upsample(128, 4),
    ]
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    last = layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=True,
        activation='tanh'
    )
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

# CHECKPOINT 652 FILE IDs (USER-PROVIDED)
FILE_IDS ={
    'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ',  # Checkpoint 652 - CORRECT
    'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe',  # Checkpoint 652 - CORRECT
    'I': '1K2DTtrsYpeB4XILn8eZAU4G6a3lty065',  # Checkpoint 652 - CORRECT
    'J': '1Reo76L5CCybAplmj_pPNWZCWFLK6n8Zp',  # Checkpoint 652 - CORRECT
}

MODELS = {}
MODELS_LOADED = False
LOADING_ERROR = None
LOADING_PROGRESS = "Starting..."

def download_and_load_models():
    global MODELS, MODELS_LOADED, LOADING_ERROR, LOADING_PROGRESS
    
    try:
        LOADING_PROGRESS = "Downloading models..."
        logger.info("=" * 70)
        logger.info("🔧 DOWNLOADING CHECKPOINT 652 MODELS (USER-PROVIDED)")
        logger.info("=" * 70)
        
        for name, file_id in FILE_IDS.items():
            LOADING_PROGRESS = f"Downloading Generator {name}..."
            logger.info(f"\n📥 Downloading Generator {name}...")
            logger.info(f"   File ID: {file_id}")
            
            output_path = f'/tmp/generator_{name.lower()}.h5'
            url = f'https://drive.google.com/uc?id={file_id}'
            
            gdown.download(url, output_path, quiet=False)
            
            if not os.path.exists(output_path):
                raise Exception(f"Generator {name} download failed - file not created")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ Generator {name} downloaded: {file_size:.2f} MB")
            
            if file_size < 10:
                raise Exception(f"Generator {name} file too small ({file_size:.2f} MB)")
            
            LOADING_PROGRESS = f"Building Generator {name}..."
            logger.info(f"🔨 Building Generator {name} architecture...")
            model = unet_generator()
            
            logger.info(f"🔧 Initializing Generator {name} layers...")
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)
            
            LOADING_PROGRESS = f"Loading weights for Generator {name}..."
            logger.info(f"📂 Loading weights for Generator {name}...")
            model.load_weights(output_path, by_name=True, skip_mismatch=True)
            
            MODELS[name] = model
            logger.info(f"✅ Generator {name} LOADED SUCCESSFULLY")
        
        MODELS_LOADED = True
        LOADING_PROGRESS = "Complete!"
        logger.info("\n" + "=" * 70)
        logger.info(f"🎉 ALL 4 CHECKPOINT 652 GENERATORS LOADED")
        logger.info("=" * 70)
        
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error("\n" + "=" * 70)
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 70)
        import traceback
        logger.error(traceback.format_exc())
        LOADING_ERROR = error_msg
        LOADING_PROGRESS = f"Failed: {error_msg}"
        MODELS_LOADED = False

logger.info("🔄 Starting model loading...")
download_and_load_models()
logger.info("✅ Model loading complete")

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
    """Convert tensor to PIL Image"""
    img_array = np.squeeze(tensor)
    img_array = (img_array + 1) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    # FIXED: Keep native 64x64 output (matches Colab training size)
    # Removed: img = img.resize((256, 256), Image.LANCZOS)  # This was causing noise!
    return img

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
        'checkpoint': '652',
        'version': 'v6.0-checkpoint-652'
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

logger.info("✅ Flask app created and ready")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
