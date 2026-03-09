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

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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

def download_from_google_drive(file_id, output_path):
    logger.info('Attempting download with requests library')
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = url + '&confirm=' + value
            response = session.get(url, stream=True)
            break
    chunk_size = 32768
    total_size = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    logger.info('Downloaded ' + str(total_size) + ' bytes')
    return total_size > 0

def initialize_models():
    global MODELS, MODELS_LOADED
    logger.info("INITIALIZING MODELS AT STARTUP")
    try:
        logger.info("Starting weight downloads from Google Drive")
        temp_dir = '/tmp/weights'
        os.makedirs(temp_dir, exist_ok=True)
        weight_paths = {}
        for name, file_id in WEIGHT_FILES.items():
            output_path = os.path.join(temp_dir, 'generator_' + name.lower() + '.h5')
            logger.info('Downloading ' + name + ' from ' + file_id)
            success = download_from_google_drive(file_id, output_path)
            if success and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(name + ' downloaded successfully ' + str(size_mb) + ' MB')
                weight_paths[name] = output_path
            else:
                logger.error(name + ' download failed')
                return False
        logger.info("Building and loading generators")
        for name in ['F', 'G', 'I', 'J']:
            logger.info('Building ' + name + ' U-Net architecture')
            generator = unet_generator()
            logger.info('Loading ' + name + ' weights from ' + weight_paths[name])
            generator.load_weights(weight_paths[name])
            MODELS[name] = generator
            logger.info(name + ' SUCCESS')
        MODELS_LOADED = True
        logger.info('MODELS LOADED: ' + str(len(MODELS)) + '/4')
        logger.info("ALL SYSTEMS READY")
        return True
    except Exception as e:
        logger.error('INITIALIZATION FAILED: ' + str(e))
        import traceback
        logger.error(traceback.format_exc())
        MODELS_LOADED = False
        return False

logger.info("STARTING I-TRANSLATION BACKEND v4.8.6")
initialize_models()

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
    return jsonify({'status': 'online', 'models_loaded': MODELS_LOADED, 'generators': {'F': 'F' in MODELS, 'G': 'G' in MODELS, 'I': 'I' in MODELS, 'J': 'J' in MODELS}, 'version': 'v4.8.6'})

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
            output_bytes = postprocess_image(output_tensor)
            results[name] = output_bytes.hex()
        return jsonify({'success': True, 'outputs': results})
    except Exception as e:
        logger.error('Conversion error: ' + str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
