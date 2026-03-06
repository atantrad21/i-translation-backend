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
    inputs = layers.Input(shape=[64, 64, 1], name=f'{name}_input')
    
    down_stack = [
        downsample(64, 4, apply_norm=False, name=f'{name}_down1'),
        downsample(128, 4, name=f'{name}_down2'),
        downsample(256, 4, name=f'{name}_down3'),
        downsample(512, 4, name=f'{name}_down4'),
    ]
    
    up_stack = [
        upsample(256, 4, apply_dropout=True, name=f'{name}_up1'),
        upsample(128, 4, apply_dropout=True, name=f'{name}_up2'),
        upsample(64, 4, name=f'{name}_up3'),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, activation='tanh',
                                 name=f'{name}_output')
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f'{up.name}_concat')([x, skip])
    
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x, name=name)

logger.info("="*80)
logger.info("🚀 I-TRANSLATION v5.2 - RAILWAY PRODUCTION")
logger.info("="*80)

# Google Drive File IDs - VERIFIED AND WORKING
GDRIVE_FILE_IDS = {
    'f': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'g': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
}

GENERATORS = {}

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using gdown with proper error handling"""
    try:
        import gdown
        logger.info(f"📥 Downloading file ID: {file_id}")
        
        # Use gdown with fuzzy=True to handle various Google Drive link formats
        url = f'https://drive.google.com/uc?id={file_id}'
        result = gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        if result and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ Downloaded successfully! Size: {file_size:.2f} MB")
            return True
        else:
            logger.error(f"❌ Download failed - file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        return False

logger.info("\n📦 Starting model download and loading process...")
logger.info(f"Total models to load: {len(GDRIVE_FILE_IDS)}")

for gen_name, file_id in GDRIVE_FILE_IDS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"[{gen_name.upper()}] Processing Generator {gen_name.upper()}")
    logger.info(f"{'='*60}")
    
    output_path = f'/tmp/generator_{gen_name}.h5'
    
    # Download file
    if download_from_gdrive(file_id, output_path):
        try:
            logger.info(f"[{gen_name.upper()}] 🏗️  Building model architecture...")
            model = unet_generator(name=f'generator_{gen_name}')
            
            logger.info(f"[{gen_name.upper()}] 🔧 Initializing layers...")
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)
            
            logger.info(f"[{gen_name.upper()}] 📂 Loading weights from file...")
            model.load_weights(output_path)
            
            logger.info(f"[{gen_name.upper()}] ✅ Model loaded successfully!")
            GENERATORS[gen_name] = model
            
            # Clean up downloaded file to save disk space
            os.remove(output_path)
            logger.info(f"[{gen_name.upper()}] 🧹 Cleaned up temporary file")
            
        except Exception as e:
            logger.error(f"[{gen_name.upper()}] ❌ Failed to load model: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
    else:
        logger.error(f"[{gen_name.upper()}] ❌ Download failed, skipping model load")

logger.info("\n" + "="*80)
logger.info(f"📊 FINAL STATUS: {len(GENERATORS)}/4 models loaded")
logger.info("="*80)

if len(GENERATORS) == 4:
    logger.info("🎉 SUCCESS! All 4 models loaded and ready!")
else:
    logger.warning(f"⚠️  WARNING: Only {len(GENERATORS)}/4 models loaded")
    logger.warning(f"Available: {list(GENERATORS.keys())}")
    logger.warning(f"Missing: {[k for k in ['f', 'g', 'i', 'j'] if k not in GENERATORS]}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = img_array[np.newaxis, :, :, np.newaxis]
    return img_array

def postprocess_image(tensor):
    img_array = tensor[0]
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    img_array = img_array[:, :, 0]
    img = Image.fromarray(img_array, mode='L')
    img = img.resize((256, 256), Image.LANCZOS)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready' if len(GENERATORS) == 4 else 'partial',
        'models_loaded': len(GENERATORS) == 4,
        'generators_available': list(GENERATORS.keys()),
        'generators_missing': [k for k in ['f', 'g', 'i', 'j'] if k not in GENERATORS],
        'total_models': len(GENERATORS),
        'version': '5.2.0',
        'checkpoint': 884,
        'platform': 'Railway'
    })

@app.route('/convert', methods=['POST'])
def convert():
    if len(GENERATORS) == 0:
        return jsonify({'error': 'No models loaded yet. Please check logs.'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    input_tensor = preprocess_image(image_bytes)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for gen_name, model in GENERATORS.items():
            output_tensor = model(input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor.numpy())
            zip_file.writestr(f'generator_{gen_name}.png', output_bytes.getvalue())
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='results.zip')

@app.route('/convert-batch', methods=['POST'])
def convert_batch():
    if len(GENERATORS) == 0:
        return jsonify({'error': 'No models loaded yet. Please check logs.'}), 503
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    images = request.files.getlist('images')
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, image_file in enumerate(images):
            image_bytes = image_file.read()
            input_tensor = preprocess_image(image_bytes)
            
            for gen_name, model in GENERATORS.items():
                output_tensor = model(input_tensor, training=False)
                output_bytes = postprocess_image(output_tensor.numpy())
                zip_file.writestr(f'image_{idx}_generator_{gen_name}.png', output_bytes.getvalue())
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='batch_results.zip')

logger.info("\n" + "="*80)
logger.info("🌐 API ENDPOINTS REGISTERED")
logger.info("="*80)
logger.info("✓ GET  /health        - Check system status")
logger.info("✓ POST /convert       - Convert single image")
logger.info("✓ POST /convert-batch - Convert multiple images")
logger.info("="*80)
logger.info("✅ APPLICATION READY TO SERVE REQUESTS")
logger.info("="*80 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
