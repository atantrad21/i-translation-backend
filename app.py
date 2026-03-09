"""
I-TRANSLATION BACKEND v4.8.2 - RENDER DEPLOYMENT
Correct 64x64 Grayscale Architecture (Checkpoint 652)
GUNICORN POST-FORK LOADING: Models load in worker processes
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
import sys

print("\n" + "="*80, flush=True)
print("I-TRANSLATION BACKEND v4.8.2 - RENDER DEPLOYMENT", flush=True)
print("Architecture: 64x64 Grayscale (Checkpoint 652)", flush=True)
print("POST-FORK LOADING: Models load in worker processes", flush=True)
print("="*80 + "\n", flush=True)

# ============================================================================
# CUSTOM LAYERS (EXACT FROM COLAB)
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
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

def downsample(filters, size, apply_norm=True, name=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=name)
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False,
                            name=f'{name}_conv' if name else None))
    if apply_norm:
        result.add(InstanceNormalization(name=f'{name}_norm' if name else None))
    result.add(layers.LeakyReLU(name=f'{name}_leaky' if name else None))
    return result

def upsample(filters, size, apply_dropout=False, name=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=name)
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False,
                                     name=f'{name}_conv' if name else None))
    result.add(InstanceNormalization(name=f'{name}_norm' if name else None))
    if apply_dropout:
        result.add(layers.Dropout(0.5, name=f'{name}_dropout' if name else None))
    result.add(layers.ReLU(name=f'{name}_relu' if name else None))
    return result

def unet_generator(output_channels=1, name='generator'):
    """
    U-Net Generator - EXACT ARCHITECTURE FROM COLAB CHECKPOINT 652
    Input: [64, 64, 1] grayscale
    Output: [64, 64, 1] grayscale
    """
    inputs = layers.Input(shape=[64, 64, 1], name=f'{name}_input')

    # Down stack: 6 layers (EXACT FROM COLAB)
    down_stack = [
        downsample(128, 4, False, name=f'{name}_down1'),
        downsample(256, 4, name=f'{name}_down2'),
        downsample(256, 4, name=f'{name}_down3'),
        downsample(256, 4, name=f'{name}_down4'),
        downsample(256, 4, name=f'{name}_down5'),
        downsample(256, 4, name=f'{name}_down6')
    ]

    # Up stack: 5 layers (EXACT FROM COLAB)
    up_stack = [
        upsample(256, 4, True, name=f'{name}_up1'),
        upsample(256, 4, True, name=f'{name}_up2'),
        upsample(256, 4, name=f'{name}_up3'),
        upsample(256, 4, name=f'{name}_up4'),
        upsample(128, 4, name=f'{name}_up5')
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, activation='tanh',
                                 name=f'{name}_output')

    concat = layers.Concatenate()

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    return keras.Model(inputs=inputs, outputs=x, name=name)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

GENERATORS = {}
MODELS_LOADED = False
GOOGLE_DRIVE_IDS = {
    'f': '1-4P7ls5G6aAjHd_LlbVXY_Sh-7lqxVWb',
    'g': '1-3QOCyAFHXs_oBbzqEiRRmXhPJOPXhBZ',
    'i': '1-2p7Cj_YLHMBPVQOTOLlx2HjXVgRgEE0',
    'j': '1-8qiZVzqwcvW3xvxYhBo2_8vvvKfYU3K'
}

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def download_weights():
    """Download weights from Google Drive using gdown"""
    print("\n[DOWNLOAD] Starting weight downloads from Google Drive...", flush=True)
    
    for gen_name, file_id in GOOGLE_DRIVE_IDS.items():
        output_path = f'/tmp/generator_{gen_name}.h5'
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[{gen_name.upper()}] ✓ Already downloaded ({file_size:.2f} MB)", flush=True)
            continue
        
        try:
            print(f"[{gen_name.upper()}] Downloading from Google Drive...", flush=True)
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[{gen_name.upper()}] ✓ Downloaded successfully ({file_size:.2f} MB)", flush=True)
            else:
                print(f"[{gen_name.upper()}] ✗ Download failed: File not created", flush=True)
                return False
                
        except Exception as e:
            print(f"[{gen_name.upper()}] ✗ Download failed: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    print("\n[DOWNLOAD] ✓ All weights downloaded successfully!\n", flush=True)
    return True

def load_models():
    """Load all 4 generator models"""
    global GENERATORS, MODELS_LOADED
    
    print("[MODELS] Building and loading generators...", flush=True)
    
    for gen_name in ['f', 'g', 'i', 'j']:
        weight_path = f'/tmp/generator_{gen_name}.h5'
        
        if not os.path.exists(weight_path):
            print(f"[{gen_name.upper()}] ✗ Weight file not found: {weight_path}", flush=True)
            continue
        
        try:
            file_size = os.path.getsize(weight_path) / (1024 * 1024)
            print(f"\n[{gen_name.upper()}] Found weights ({file_size:.2f} MB)", flush=True)
            
            print(f"[{gen_name.upper()}] Building 64x64 grayscale architecture...", flush=True)
            model = unet_generator(name=f'generator_{gen_name}')
            
            print(f"[{gen_name.upper()}] Initializing layers...", flush=True)
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)
            
            print(f"[{gen_name.upper()}] Loading weights...", flush=True)
            model.load_weights(weight_path)
            
            print(f"[{gen_name.upper()}] ✓ SUCCESS!", flush=True)
            GENERATORS[gen_name] = model
            
        except Exception as e:
            print(f"[{gen_name.upper()}] ✗ FAILED: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80, flush=True)
    print(f"MODELS LOADED: {len(GENERATORS)}/4", flush=True)
    print("="*80 + "\n", flush=True)
    
    MODELS_LOADED = (len(GENERATORS) == 4)
    return MODELS_LOADED

def initialize_models():
    """Initialize models - called by worker processes"""
    global MODELS_LOADED
    
    if MODELS_LOADED:
        print("[WORKER] Models already loaded, skipping...", flush=True)
        return True
    
    print(f"[WORKER] Worker PID: {os.getpid()}", flush=True)
    print("[WORKER] Initializing models in worker process...", flush=True)
    
    if download_weights():
        if load_models():
            print("[WORKER] ✓ ALL SYSTEMS READY!", flush=True)
            return True
        else:
            print("[WORKER] ✗ Model loading failed!", flush=True)
            return False
    else:
        print("[WORKER] ✗ Weight download failed!", flush=True)
        return False

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def preprocess_image(image_bytes):
    """Convert input image to 64x64 grayscale tensor"""
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = img_array[np.newaxis, :, :, np.newaxis]
    return tf.constant(img_array)

def postprocess_image(output_tensor, upscale_to=256):
    """Convert output tensor to image bytes (upscaled for display)"""
    output_array = output_tensor.numpy()[0]
    output_array = ((output_array + 1.0) * 127.5).astype(np.uint8)
    output_array = output_array[:, :, 0]
    
    img = Image.fromarray(output_array, mode='L')
    
    if upscale_to:
        img = img.resize((upscale_to, upscale_to), Image.LANCZOS)
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Try to initialize models if not loaded
    if not MODELS_LOADED:
        print("[HEALTH] Models not loaded, attempting to initialize...", flush=True)
        initialize_models()
    
    return jsonify({
        'status': 'online',
        'version': '4.8.2',
        'architecture': '64x64 grayscale',
        'checkpoint': 652,
        'models_loaded': MODELS_LOADED,
        'generators': {
            'f': 'f' in GENERATORS,
            'g': 'g' in GENERATORS,
            'i': 'i' in GENERATORS,
            'j': 'j' in GENERATORS
        }
    })

@app.route('/convert', methods=['POST'])
def convert():
    """Convert CT to MRI or MRI to CT"""
    # Try to initialize models if not loaded
    if not MODELS_LOADED:
        print("[CONVERT] Models not loaded, attempting to initialize...", flush=True)
        initialize_models()
    
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        input_tensor = preprocess_image(image_bytes)
        
        results = {}
        for gen_name in ['f', 'g', 'i', 'j']:
            output_tensor = GENERATORS[gen_name](input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor, upscale_to=256)
            
            import base64
            output_b64 = base64.b64encode(output_bytes).decode('utf-8')
            results[f'generator_{gen_name}'] = output_b64
        
        return jsonify({
            'success': True,
            'results': results,
            'info': {
                'input_size': '64x64 grayscale',
                'output_size': '256x256 grayscale (upscaled)',
                'checkpoint': 652
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'I-Translation Backend',
        'version': '4.8.2',
        'architecture': '64x64 grayscale',
        'checkpoint': 652,
        'status': 'online' if MODELS_LOADED else 'loading',
        'endpoints': {
            'health': '/health',
            'convert': '/convert (POST with image file)'
        }
    })

# ============================================================================
# GUNICORN HOOKS (FOR PRODUCTION)
# ============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    print("[GUNICORN] Master process starting...", flush=True)

def when_ready(server):
    """Called just after the server is started."""
    print("[GUNICORN] Server is ready", flush=True)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"\n[GUNICORN] Worker {worker.pid} forked", flush=True)
    print(f"[GUNICORN] Initializing models in worker {worker.pid}...", flush=True)
    initialize_models()

# ============================================================================
# DIRECT EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n[DIRECT] Running in direct execution mode", flush=True)
    initialize_models()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
