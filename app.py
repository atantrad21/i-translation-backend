from flask import Flask, request, jsonify, send_file
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB

# Global model storage
GENERATORS = {}

# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def downsample(filters, size, apply_norm=True, name_prefix="down"):
    """Downsampling block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=f"{name_prefix}_seq")
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False,
                            name=f"{name_prefix}_conv"))
    if apply_norm:
        result.add(InstanceNormalization(name=f"{name_prefix}_norm"))
    result.add(layers.LeakyReLU(name=f"{name_prefix}_relu"))
    return result

def upsample(filters, size, apply_dropout=False, name_prefix="up"):
    """Upsampling block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=f"{name_prefix}_seq")
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False,
                                     name=f"{name_prefix}_convT"))
    result.add(InstanceNormalization(name=f"{name_prefix}_norm"))
    if apply_dropout:
        result.add(layers.Dropout(0.5, name=f"{name_prefix}_drop"))
    result.add(layers.ReLU(name=f"{name_prefix}_relu"))
    return result

def unet_generator(name="generator"):
    """U-Net Generator"""
    inputs = layers.Input(shape=[256, 256, 3], name=f"{name}_input")
    
    down_stack = [
        downsample(64, 4, apply_norm=False, name_prefix=f"{name}_down1"),
        downsample(128, 4, name_prefix=f"{name}_down2"),
        downsample(256, 4, name_prefix=f"{name}_down3"),
        downsample(512, 4, name_prefix=f"{name}_down4"),
        downsample(512, 4, name_prefix=f"{name}_down5"),
        downsample(512, 4, name_prefix=f"{name}_down6"),
        downsample(512, 4, name_prefix=f"{name}_down7"),
        downsample(512, 4, name_prefix=f"{name}_down8"),
    ]
    
    up_stack = [
        upsample(512, 4, apply_dropout=True, name_prefix=f"{name}_up1"),
        upsample(512, 4, apply_dropout=True, name_prefix=f"{name}_up2"),
        upsample(512, 4, apply_dropout=True, name_prefix=f"{name}_up3"),
        upsample(512, 4, name_prefix=f"{name}_up4"),
        upsample(256, 4, name_prefix=f"{name}_up5"),
        upsample(128, 4, name_prefix=f"{name}_up6"),
        upsample(64, 4, name_prefix=f"{name}_up7"),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                 kernel_initializer=initializer,
                                 activation='tanh',
                                 name=f"{name}_output_conv")
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f"{name}_concat_{up.name}")([x, skip])
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x, name=name)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Download and load all 4 generator models"""
    logger.info("="*80)
    logger.info("I-TRANSLATION v4.7.6 - BUILD ARCHITECTURE + LOAD WEIGHTS")
    logger.info("="*80)
    
    # FIXED: Added missing "1" prefix to Generator G file ID
    model_configs = {
        'F': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
        'G': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',  # ✅ FIXED
        'I': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
        'J': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
    }
    
    logger.info(f"Starting model loading process...")
    logger.info(f"Total models to load: {len(model_configs)}")
    
    for gen_name, file_id in model_configs.items():
        logger.info("="*60)
        logger.info(f"[{gen_name}] Processing Generator {gen_name}")
        logger.info("="*60)
        
        weight_path = f"/tmp/generator_{gen_name.lower()}.h5"
        
        try:
            # Step 1: Download weights
            logger.info(f"Downloading file ID: {file_id}")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, weight_path, quiet=False)
            
            file_size = os.path.getsize(weight_path) / (1024 * 1024)
            logger.info(f"Downloaded successfully! Size: {file_size:.2f} MB")
            
            # Step 2: Build architecture
            logger.info(f"[{gen_name}] Building U-Net architecture...")
            model = unet_generator(name=f"generator_{gen_name.lower()}")
            
            # Step 3: Initialize with dummy input
            logger.info(f"[{gen_name}] Initializing layers with dummy input...")
            dummy_input = tf.zeros((1, 256, 256, 3))
            _ = model(dummy_input, training=False)
            
            # Step 4: Load weights
            logger.info(f"[{gen_name}] Loading weights into architecture...")
            model.load_weights(weight_path, by_name=True, skip_mismatch=False)
            
            # Step 5: Store model
            GENERATORS[gen_name] = model
            logger.info(f"[{gen_name}] ✅ Weights loaded successfully!")
            
            # Cleanup
            os.remove(weight_path)
            logger.info(f"[{gen_name}] Cleaned up weight file")
            
        except Exception as e:
            logger.error(f"[{gen_name}] ❌ Error loading model: {str(e)}")
            GENERATORS[gen_name] = None
    
    logger.info("")
    logger.info("="*80)
    loaded_count = sum(1 for v in GENERATORS.values() if v is not None)
    logger.info(f"✅ {loaded_count}/{len(model_configs)} MODELS LOADED SUCCESSFULLY")
    logger.info("="*80)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256), Image.LANCZOS)
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

def postprocess_image(prediction):
    """Convert model output to image"""
    prediction = (prediction[0] + 1.0) * 127.5
    prediction = np.clip(prediction, 0, 255).astype(np.uint8)
    return Image.fromarray(prediction)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = {k: (v is not None) for k, v in GENERATORS.items()}
    all_loaded = all(models_loaded.values())
    
    return jsonify({
        'status': 'healthy' if all_loaded else 'degraded',
        'models_loaded': models_loaded,
        'version': 'v4.7.6-FIXED'
    })

@app.route('/convert', methods=['POST'])
def convert_image():
    """Convert images using all 4 generators"""
    try:
        if not all(GENERATORS.values()):
            return jsonify({'error': 'Models not loaded'}), 503
        
        conversion_type = request.form.get('type', 'ct_to_mri')
        
        # Process up to 4 images
        results = []
        for i in range(1, 5):
            file_key = f'image{i}'
            if file_key not in request.files:
                continue
            
            file = request.files[file_key]
            if file.filename == '':
                continue
            
            # Read and preprocess
            image_bytes = file.read()
            input_tensor = preprocess_image(image_bytes)
            
            # Convert with all 4 generators
            outputs = {}
            for gen_name, model in GENERATORS.items():
                if model is not None:
                    prediction = model(input_tensor, training=False)
                    output_img = postprocess_image(prediction.numpy())
                    
                    # Convert to bytes
                    img_io = io.BytesIO()
                    output_img.save(img_io, 'PNG')
                    img_io.seek(0)
                    
                    outputs[gen_name] = img_io.getvalue()
            
            results.append({
                'input_index': i,
                'outputs': outputs
            })
        
        if not results:
            return jsonify({'error': 'No valid images provided'}), 400
        
        # Return first result's outputs (simplified for now)
        if results:
            first_result = results[0]['outputs']
            return jsonify({
                'success': True,
                'message': f'Converted {len(results)} image(s)',
                'generators_used': list(first_result.keys())
            })
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STARTUP
# ============================================================================

# Load models at startup (module level for gunicorn)
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
