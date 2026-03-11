"""
I-Translation Medical Image Converter - Backend API
Version: v11.2 - Universal Keras compatibility patch
Checkpoint: 652
Works with TensorFlow 2.4+ and Python 3.8-3.9
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import io
import requests
import base64

print("[INFO] Applying Universal Keras compatibility patch...")
print(f"[INFO] Python version: {os.sys.version}")

import tensorflow as tf
print(f"[INFO] TensorFlow version: {tf.__version__}")

# Universal patch that works with both old and new Keras
try:
    # Try new Keras (TF 2.16+)
    from keras.engine.input_layer import InputLayer
    print("[INFO] Using standalone Keras")
except ImportError:
    try:
        # Try TF integrated Keras (TF 2.4-2.15)
        from tensorflow.python.keras.engine.input_layer import InputLayer
        print("[INFO] Using TensorFlow integrated Keras")
    except ImportError:
        # Fallback to tf.keras
        from tensorflow.keras.layers import InputLayer
        print("[INFO] Using tf.keras.layers")

# Patch InputLayer to handle batch_shape and type_spec parameters
original_input_init = InputLayer.__init__

def patched_input_init(self, input_shape=None, batch_size=None, dtype=None,
                       input_tensor=None, sparse=None, ragged=None, 
                       type_spec=None, name=None, **kwargs):
    """
    Universal patch for InputLayer that handles both batch_shape and type_spec.
    Works with TensorFlow 2.4+ and Python 3.8-3.9.
    """
    
    # Remove type_spec if present (not supported in older TF)
    if type_spec is not None:
        print(f"[PATCH] Intercepted type_spec parameter")
        # Ignore it - don't pass to original init
    
    # Handle batch_shape parameter
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        print(f"[PATCH] Intercepted batch_shape: {batch_shape}")
        
        if input_shape is None and batch_shape is not None:
            input_shape = batch_shape[1:]
            if batch_size is None:
                batch_size = batch_shape[0]
    
    # Remove any other unsupported kwargs
    unsupported = ['type_spec', 'batch_shape']
    for key in unsupported:
        if key in kwargs:
            removed = kwargs.pop(key)
            print(f"[PATCH] Removed unsupported kwarg: {key}={removed}")
    
    # Build clean kwargs for original init
    clean_kwargs = {
        'input_shape': input_shape,
        'batch_size': batch_size,
        'dtype': dtype,
        'name': name,
    }
    
    # Only add optional parameters if they're not None
    if input_tensor is not None:
        clean_kwargs['input_tensor'] = input_tensor
    if sparse is not None:
        clean_kwargs['sparse'] = sparse
    if ragged is not None:
        clean_kwargs['ragged'] = ragged
    
    # Add any remaining kwargs
    clean_kwargs.update(kwargs)
    
    # Call original init with clean kwargs
    try:
        return original_input_init(self, **clean_kwargs)
    except TypeError as e:
        # If still fails, try minimal kwargs
        print(f"[PATCH] Fallback to minimal kwargs due to: {e}")
        minimal_kwargs = {
            'input_shape': input_shape,
            'batch_size': batch_size,
            'dtype': dtype,
            'name': name,
        }
        return original_input_init(self, **minimal_kwargs)

InputLayer.__init__ = patched_input_init
print("[INFO] Universal compatibility patch applied successfully!")

from tensorflow.keras import layers

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

def load_models():
    print("\n" + "="*70)
    print("[INFO] CHECKPOINT 652 LOADER (Universal Compatibility)")
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
            
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url, stream=True, timeout=600)
            
            if response.status_code != 200:
                print(f"[ERROR] Download failed: HTTP {response.status_code}")
                continue
            
            model_file = f"/tmp/generator_{name.lower()}.h5"
            with open(model_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            print(f"[INFO] Downloaded: {file_size:.1f} MB")
            
            print(f"[INFO] Loading model with patched Keras...")
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
            os.remove(model_file)
                
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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

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
        'python_version': os.sys.version.split()[0],
        'version': 'v11.2-universal-patch'
    })

@app.route('/convert', methods=['POST'])
def convert_image():
    if len(generators) != 4:
        return jsonify({'error': 'Models not loaded yet, please wait'}), 503
    
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
            
            # Select generators based on conversion type
            if conversion_type == 'ct_to_mri':
                # CT to MRI: Use generators G and I
                gen_outputs = {}
                if 'G' in generators:
                    gen_outputs['G'] = generators['G'](input_tensor, training=False)
                if 'I' in generators:
                    gen_outputs['I'] = generators['I'](input_tensor, training=False)
            else:  # mri_to_ct
                # MRI to CT: Use generators F and J
                gen_outputs = {}
                if 'F' in generators:
                    gen_outputs['F'] = generators['F'](input_tensor, training=False)
                if 'J' in generators:
                    gen_outputs['J'] = generators['J'](input_tensor, training=False)
            
            # Convert all generator outputs to images
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
