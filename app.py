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

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config()
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
    result.add(layers.LeakyReLU(name=f'{name}_activation' if name else None))
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
    result.add(layers.ReLU(name=f'{name}_activation' if name else None))
    return result

def unet_generator(name='generator'):
    inputs = layers.Input(shape=[64, 64, 1], name=f'{name}_input')
    
    down_stack = [
        downsample(64, 4, apply_norm=False, name=f'{name}_down1'),
        downsample(128, 4, name=f'{name}_down2'),
        downsample(256, 4, name=f'{name}_down3'),
        downsample(512, 4, name=f'{name}_down4'),
        downsample(512, 4, name=f'{name}_down5'),
        downsample(512, 4, name=f'{name}_down6'),
        downsample(512, 4, name=f'{name}_down7'),
        downsample(512, 4, name=f'{name}_down8'),
    ]
    
    up_stack = [
        upsample(512, 4, apply_dropout=True, name=f'{name}_up1'),
        upsample(512, 4, apply_dropout=True, name=f'{name}_up2'),
        upsample(512, 4, apply_dropout=True, name=f'{name}_up3'),
        upsample(512, 4, name=f'{name}_up4'),
        upsample(256, 4, name=f'{name}_up5'),
        upsample(128, 4, name=f'{name}_up6'),
        upsample(64, 4, name=f'{name}_up7'),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same',
                                 kernel_initializer=initializer,
                                 activation='tanh',
                                 name=f'{name}_output')
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f'{name}_concat_{up.name}')([x, skip])
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x, name=name)

GENERATORS = {}

GDRIVE_FILE_IDS = {
    'f': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'g': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
}

def download_from_gdrive_requests(file_id, output_path):
    """Download file from Google Drive using requests library (more reliable)"""
    try:
        logger.info(f"📥 Downloading file ID: {file_id}")
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle the confirmation token for large files
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                response = session.get(url, stream=True)
                break
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
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
            logger.info(f"✅ Downloaded successfully! Size: {file_size:.2f} MB")
            return True
        else:
            logger.error(f"❌ Download failed - file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        return False

logger.info("="*80)
logger.info("🚀 I-TRANSLATION v5.3 - RAILWAY PRODUCTION (FIXED DOWNLOAD)")
logger.info("="*80)
logger.info("")
logger.info("📦 Starting model download and loading process...")
logger.info(f"Total models to load: {len(GDRIVE_FILE_IDS)}")
logger.info("")

for gen_name, file_id in GDRIVE_FILE_IDS.items():
    logger.info("="*60)
    logger.info(f"[{gen_name.upper()}] Processing Generator {gen_name.upper()}")
    logger.info("="*60)
    
    output_path = f'/tmp/generator_{gen_name}.h5'
    
    if download_from_gdrive_requests(file_id, output_path):
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
            
            os.remove(output_path)
            logger.info(f"[{gen_name.upper()}] 🧹 Cleaned up temporary file")
            
        except Exception as e:
            logger.error(f"[{gen_name.upper()}] ❌ Error loading model: {str(e)}")
    else:
        logger.error(f"[{gen_name.upper()}] ❌ Download failed, skipping model load")
    
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
    img_array = tensor<sup>0</sup>
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    img_array = img_array[:, :, 0]
    img = Image.fromarray(img_array, mode='L')
    return img

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(GENERATORS) == 4,
        'loaded_models': list(GENERATORS.keys())
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
logger.info("")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
