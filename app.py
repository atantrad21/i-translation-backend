"""
I-Translation Medical Image Converter - Backend API
Version: v12.1 - TensorFlow 2.10+ (Clean, No Patches Needed)
Checkpoint: 652
Native compatibility with models saved in TensorFlow 2.10+
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown
import numpy as np
from PIL import Image
import os
import sys
import io
import requests
import base64
import tensorflow as tf
from tensorflow.keras import layers

print(f"[INFO] Python version: {sys.version}")
print(f"[INFO] TensorFlow version: {tf.__version__}")
print("[INFO] Using TensorFlow 2.16.1 (Keras 3) for native Checkpoint 652 compatibility")

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
        config.update({'epsilon': self.epsilon})
        return config

def download_from_gdrive(file_id, destination):
    """Safely downloads large files from Google Drive using gdown."""
    try:
        # gdown automatically handles the virus scan confirmation
        gdown.download(id=file_id, output=destination, quiet=False)
        
        # Verify the file actually downloaded and isn't empty
        if os.path.exists(destination) and os.path.getsize(destination) > 10000:
            return True
        return False
    except Exception as e:
        print(f"[ERROR] gdown failed: {str(e)}")
        return False
    # Check for the confirmation token in cookies
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        return True
    return False

def load_models():
    print("\n" + "="*70)
    print("[INFO] CHECKPOINT 652 LOADER (TensorFlow 2.10+ Native)")
    print("="*70)
    
    file_ids = {
        'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ',
        'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe',
        'I': '1K2DTtrsYpeB4XILn8eZAU4G6a3lty065',
        'J': '1Reo76L5CCybAplmj_pPNWZCWFLK6n8Zp'
    }
    
    generators = {}
    
    for name, file_id in file_ids.items():
        try:
            print(f"\n[INFO] Processing Generator {name}...")
            print(f"[INFO] Downloading from Google Drive...")
            
            model_file = f"/tmp/generator_{name.lower()}.h5"
            
            # Use the robust download function
            success = download_from_gdrive(file_id, model_file)
            
            if not success:
                print(f"[ERROR] Download failed for Generator {name}.")
                continue
            
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            print(f"[INFO] Downloaded: {file_size:.1f} MB")
            
            print(f"[INFO] Loading model with TensorFlow {tf.__version__}...")
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={'InstanceNormalization': InstanceNormalization},
                compile=False
            )
            
            print(f"[SUCCESS] Generator {name} LOADED!")
            print(f"          Layers: {len(model.layers)}")
            print(f"          Input: {model.input_shape}")
            print(f"          Output: {model.output_shape}")
            
            generators[name] = model
            os.remove(model_file) # Clean up after loading to save space
                
        except Exception as e:
            print(f"[ERROR] Generator {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    if len(generators) == 4:
        print("🎉 ALL 4 CHECKPOINT 652 GENERATORS LOADED!")
    else:
        print(f"⚠️  Only {len(generators)}/4 generators loaded")
    print("="*70 + "\n")
    
    return generators

print("[INFO] Starting module initialization...")
generators = load_models()
print(f"[INFO] Initialization complete. Loaded: {len(generators)}/4 generators\n")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024 # 10GB limit

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(prediction):
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=-1)
    prediction = ((prediction + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(prediction, mode='L')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'models_loaded': len(generators) == 4,
        'generators': list(generators.keys()),
        'checkpoint': '652',
        'tensorflow_version': tf.__version__,
        'python_version': sys.version.split()[0],
        'version': 'v12.1-tensorflow-2.10-native'
    })

@app.route('/convert', methods=['POST'])
def convert_image():
    if len(generators) != 4:
        return jsonify({'error': 'Models not loaded yet or failed to load, please check server logs.'}), 503
    
    conversion_type = request.form.get('type', 'ct_to_mri')
    results = {}
    
    for i in range(1, 5):
        image_key = f'image{i}'
        if image_key not in request.files:
            continue
            
        try:
            image_file = request.files[image_key]
            image_bytes = image_file.read()
            input_tensor = preprocess_image(image_bytes)
            
            gen_outputs = {}
            if conversion_type == 'ct_to_mri':
                if 'G' in generators:
                    gen_outputs['G'] = generators['G'](input_tensor, training=False)
                if 'I' in generators:
                    gen_outputs['I'] = generators['I'](input_tensor, training=False)
            else:
                if 'F' in generators:
                    gen_outputs['F'] = generators['F'](input_tensor, training=False)
                if 'J' in generators:
                    gen_outputs['J'] = generators['J'](input_tensor, training=False)
            
            for gen_name, prediction in gen_outputs.items():
                output_img = postprocess_image(prediction.numpy())
                
                img_byte_arr = io.BytesIO()
                output_img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                results[f'{image_key}_{gen_name}'] = img_base64
            
        except Exception as e:
            print(f"[ERROR] Processing {image_key}: {str(e)}")
            results[f'{image_key}_error'] = str(e)
    
    if not results:
        return jsonify({'error': 'No valid images provided'}), 400
    
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
