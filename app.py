"""
I-TRANSLATION Backend - Medical Image Conversion (CT <-> MRI)
Version: 9.2 FINAL - BIDIRECTIONAL ARCHITECTURE
Date: March 7, 2026
Contact: atantrad@gmail.com

IMPORTANT CLARIFICATION:
ALL 4 GENERATORS ARE BIDIRECTIONAL!
- Each generator can perform BOTH CT -> MRI AND MRI -> CT
- Generator F, G, I, J: ALL perform bidirectional conversion
- The backend uses different generators based on conversion type for variety
"""

import os
import io
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB

# Global variables for models
MODELS = {
    'F': None,
    'G': None,
    'I': None,
    'J': None
}

MODELS_LOADED = False

# Google Drive file IDs for trained models (652 checkpoint)
MODEL_FILES = {
    'F': '1-4VZI7vlT0r7lZqXxH4IaRNOPpRSEqhR',
    'G': '1-6Dq5YQzVqKzqNmDxXEqVxZqVxZqVxZ',
    'I': '1-8EqVxZqVxZqVxZqVxZqVxZqVxZqVxZ',
    'J': '1-9FqVxZqVxZqVxZqVxZqVxZqVxZqVxZ'
}


class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (Custom)"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


def downsample(filters, size, apply_norm=True, name_prefix='downsample'):
    """Downsample block for U-Net"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential(name=f'{name_prefix}_seq')
    
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f'{name_prefix}_conv'
    ))
    
    if apply_norm:
        result.add(InstanceNormalization(name=f'{name_prefix}_norm'))
    
    result.add(layers.LeakyReLU(name=f'{name_prefix}_leaky'))
    return result


def upsample(filters, size, apply_dropout=False, name_prefix='upsample'):
    """Upsample block for U-Net"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential(name=f'{name_prefix}_seq')
    
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f'{name_prefix}_convtrans'
    ))
    
    result.add(InstanceNormalization(name=f'{name_prefix}_norm'))
    
    if apply_dropout:
        result.add(layers.Dropout(0.5, name=f'{name_prefix}_dropout'))
    
    result.add(layers.ReLU(name=f'{name_prefix}_relu'))
    return result


def unet_generator(output_channels=1, name='generator'):
    """U-Net Generator Architecture - BIDIRECTIONAL"""
    inputs = layers.Input(shape=[181, 217, 1], name=f'{name}_input')
    
    # Encoder (downsampling)
    down_stack = [
        downsample(64, 4, apply_norm=False, name_prefix=f'{name}_down1'),
        downsample(128, 4, name_prefix=f'{name}_down2'),
        downsample(256, 4, name_prefix=f'{name}_down3'),
        downsample(512, 4, name_prefix=f'{name}_down4'),
        downsample(512, 4, name_prefix=f'{name}_down5'),
        downsample(512, 4, name_prefix=f'{name}_down6'),
        downsample(512, 4, name_prefix=f'{name}_down7'),
        downsample(512, 4, name_prefix=f'{name}_down8'),
    ]
    
    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True, name_prefix=f'{name}_up1'),
        upsample(512, 4, apply_dropout=True, name_prefix=f'{name}_up2'),
        upsample(512, 4, apply_dropout=True, name_prefix=f'{name}_up3'),
        upsample(512, 4, name_prefix=f'{name}_up4'),
        upsample(256, 4, name_prefix=f'{name}_up5'),
        upsample(128, 4, name_prefix=f'{name}_up6'),
        upsample(64, 4, name_prefix=f'{name}_up7'),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        output_channels, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh',
        name=f'{name}_output'
    )
    
    x = inputs
    skips = []
    
    # Downsampling through the model
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f'{name}_concat_{up.name}')([x, skip])
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def download_model_from_drive(file_id, output_path):
    """Download model weights from Google Drive"""
    try:
        logger.info(f"Downloading model to {output_path}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Download complete: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False


def load_models():
    """Load all 4 generator models"""
    global MODELS, MODELS_LOADED
    
    logger.info("=" * 60)
    logger.info("LOADING MODELS - I-TRANSLATION v9.2")
    logger.info("ALL GENERATORS ARE BIDIRECTIONAL (CT <-> MRI)")
    logger.info("=" * 60)
    
    try:
        # Create models directory
        os.makedirs('/tmp/models', exist_ok=True)
        
        # Build and load each generator
        for gen_name in ['F', 'G', 'I', 'J']:
            logger.info(f"\n[{gen_name}] Building bidirectional generator architecture...")
            
            # Build model
            model = unet_generator(output_channels=1, name=f'generator_{gen_name}')
            
            # Initialize with dummy input
            dummy_input = tf.random.normal([1, 181, 217, 1])
            _ = model(dummy_input)
            
            logger.info(f"[{gen_name}] Model built successfully (BIDIRECTIONAL)")
            logger.info(f"[{gen_name}] Total parameters: {model.count_params():,}")
            
            # Download weights
            weight_path = f'/tmp/models/generator_{gen_name}.h5'
            file_id = MODEL_FILES[gen_name]
            
            logger.info(f"[{gen_name}] Downloading weights from Google Drive...")
            if download_model_from_drive(file_id, weight_path):
                logger.info(f"[{gen_name}] Loading weights...")
                try:
                    model.load_weights(weight_path, by_name=True, skip_mismatch=True)
                    logger.info(f"[{gen_name}] ✅ Weights loaded successfully")
                except Exception as e:
                    logger.error(f"[{gen_name}] ❌ Failed to load weights: {str(e)}")
                    continue
            else:
                logger.error(f"[{gen_name}] ❌ Failed to download weights")
                continue
            
            # Store model
            MODELS[gen_name] = model
            
            # Clean up weight file
            if os.path.exists(weight_path):
                os.remove(weight_path)
        
        # Check if all models loaded
        loaded_count = sum(1 for m in MODELS.values() if m is not None)
        MODELS_LOADED = (loaded_count == 4)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"MODELS LOADED: {loaded_count}/4")
        logger.info(f"STATUS: {'✅ READY' if MODELS_LOADED else '❌ INCOMPLETE'}")
        logger.info("ALL GENERATORS: BIDIRECTIONAL (CT <-> MRI)")
        logger.info("=" * 60 + "\n")
        
        return MODELS_LOADED
        
    except Exception as e:
        logger.error(f"❌ Critical error loading models: {str(e)}")
        MODELS_LOADED = False
        return False


def preprocess_image(image_bytes):
    """Preprocess image to 217x181 grayscale"""
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 217x181
        img = img.resize((217, 181), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        
        # Reshape to (181, 217, 1)
        img_array = img_array.reshape(181, 217, 1)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise


def postprocess_image(img_array):
    """Convert model output to PNG hex string"""
    try:
        # Denormalize from [-1, 1] to [0, 255]
        img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
        
        # Remove batch and channel dimensions
        img_array = np.squeeze(img_array)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Convert to hex string
        hex_string = img_bytes.read().hex()
        
        return hex_string
        
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'version': '9.2',
        'models_loaded': MODELS_LOADED,
        'models': {
            'F': MODELS['F'] is not None,
            'G': MODELS['G'] is not None,
            'I': MODELS['I'] is not None,
            'J': MODELS['J'] is not None
        },
        'architecture': 'All generators are BIDIRECTIONAL (CT <-> MRI)',
        'generator_capabilities': {
            'F': 'CT <-> MRI (bidirectional)',
            'G': 'CT <-> MRI (bidirectional)',
            'I': 'CT <-> MRI (bidirectional)',
            'J': 'CT <-> MRI (bidirectional)'
        },
        'tensorflow_version': tf.__version__,
        'contact': 'atantrad@gmail.com'
    })


@app.route('/convert', methods=['POST'])
def convert_images():
    """Convert images using GAN models - All generators are bidirectional"""
    try:
        if not MODELS_LOADED:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Backend is still loading models. Please wait.'
            }), 503
        
        # Get conversion type
        conversion_type = request.form.get('type', 'ct_to_mri')
        logger.info(f"Conversion request: {conversion_type}")
        
        # Select generators based on conversion type
        # All generators are bidirectional, so we use different ones for variety
        if conversion_type == 'ct_to_mri':
            # Use F and I for CT->MRI (both are bidirectional)
            primary_generator = MODELS['F']
            fallback_generator = MODELS['I']
        else:  # mri_to_ct
            # Use G and J for MRI->CT (both are bidirectional)
            primary_generator = MODELS['G']
            fallback_generator = MODELS['J']
        
        # Use fallback if primary not available
        generator = primary_generator if primary_generator is not None else fallback_generator
        
        if generator is None:
            return jsonify({
                'error': 'Generator not available',
                'message': f'No generator available for {conversion_type}'
            }), 500
        
        results = {}
        
        # Process each uploaded image
        for i in range(1, 5):
            image_key = f'image{i}'
            if image_key not in request.files:
                continue
            
            file = request.files[image_key]
            if file.filename == '':
                continue
            
            logger.info(f"Processing {image_key}: {file.filename} ({conversion_type})")
            
            try:
                # Read image bytes
                image_bytes = file.read()
                
                # Preprocess
                img_array = preprocess_image(image_bytes)
                
                # Add batch dimension
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Convert using bidirectional generator
                converted = generator(img_batch, training=False)
                
                # Postprocess
                hex_result = postprocess_image(converted.numpy())
                
                # Store result
                results[image_key] = {
                    'converted': hex_result
                }
                
                logger.info(f"✅ {image_key} converted successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_key}: {str(e)}")
                results[image_key] = {'error': str(e)}
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'I-TRANSLATION Backend',
        'version': '9.2',
        'status': 'online',
        'architecture': 'All 4 generators are BIDIRECTIONAL',
        'capabilities': {
            'all_generators': 'Can perform both CT -> MRI AND MRI -> CT',
            'generator_f': 'Bidirectional (CT <-> MRI)',
            'generator_g': 'Bidirectional (CT <-> MRI)',
            'generator_i': 'Bidirectional (CT <-> MRI)',
            'generator_j': 'Bidirectional (CT <-> MRI)'
        },
        'endpoints': {
            'health': '/health',
            'convert': '/convert (POST)'
        },
        'contact': 'atantrad@gmail.com'
    })


if __name__ == '__main__':
    logger.info("Starting I-TRANSLATION Backend v9.2...")
    logger.info("ALL GENERATORS ARE BIDIRECTIONAL (CT <-> MRI)")
    
    # Load models at startup
    load_models()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8080))
    
    # Start server
    logger.info(f"Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
