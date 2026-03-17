"""
I-Translation Medical Image Converter - Backend API
Version: v13.0 - Optimized Single-Image Pipeline
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import sys
import io
import base64
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras import layers

print(f"[INFO] Python version: {sys.version}")
print(f"[INFO] TensorFlow version: {tf.__version__}")
print("[INFO] Using Optimized CycleGAN Pipeline (G & F only)")

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],),
            initializer='ones', trainable=True
        )
        self.offset = self.add_weight(
            name='offset', shape=(input_shape[-1],),
            initializer='zeros', trainable=True
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
    try:
        print(f"[INFO] Using gdown to download file...")
        gdown.download(id=file_id, output=destination, quiet=False)
        if os.path.exists(destination) and os.path.getsize(destination) > 10000:
            return True
        return False
    except Exception as e:
        print(f"[ERROR] gdown failed: {str(e)}")
        return False

def load_models():
    print("\n" + "="*70)
    print("[INFO] LOADING PRIMARY GENERATORS ONLY")
    print("="*70)
    
    # We isolated the best models: G for CT->MRI, F for MRI->CT
    file_ids = {
        'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe', 
        'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ'
    }
    
    generators = {}
    
    for name, file_id in file_ids.items():
        try:
            print(f"\n[INFO] Processing Generator {name}...")
            model_file = f"/tmp/generator_{name.lower()}.h5"
            
            if not download_from_gdrive(file_id, model_file):
                print(f"[ERROR] Download failed for Generator {name}.")
                continue
            
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={'InstanceNormalization': InstanceNormalization},
                compile=False
            )
            
            print(f"[SUCCESS] Generator {name} LOADED!")
            generators[name] = model
            os.remove(model_file)
                
        except Exception as e:
            print(f"[ERROR] Generator {name} failed: {str(e)}")
    
    return generators

generators = load_models()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def preprocess_image(image_bytes):
    # 1. Load the image and convert to grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # 2. BUMP RESOLUTION: Scale up to 256x256 for high-quality CycleGAN output
    img = img.resize((256, 256), Image.LANCZOS)
    
    # 3. Convert to a NumPy array for OpenCV processing
    img_array = np.array(img, dtype=np.uint8)
    
    # 4. APPLY CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)
    
    # 5. Normalize the image array for the AI (-1.0 to 1.0)
    img_array = img_array.astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    # 6. Expand dimensions so it matches the (1, 256, 256, 1) shape the AI expects
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
        'generators': list(generators.keys()),
        'models_loaded': len(generators) == 2
    })

@app.route('/convert', methods=['POST'])
def convert_image():
    if len(generators) != 2:
        return jsonify({'error': 'Models not fully loaded.'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    conversion_type = request.form.get('type', 'ct_to_mri')
    
    try:
        image_bytes = request.files['image'].read()
        input_tensor = preprocess_image(image_bytes)
        
        # Use G for CT->MRI, and F for MRI->CT
        if conversion_type == 'ct_to_mri':
            prediction = generators['G'](input_tensor, training=False)
        else:
            prediction = generators['F'](input_tensor, training=False)
        
        output_img = postprocess_image(prediction.numpy())
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return jsonify({'translated_image': img_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
