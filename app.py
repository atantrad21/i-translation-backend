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
import pydicom
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

def preprocess_image(image_bytes, filename, conversion_type):
    # 1. Check if the file is a DICOM or a standard image
    if filename.lower().endswith('.dcm'):
        # Decode raw DICOM medical data
        dicom_data = pydicom.dcmread(io.BytesIO(image_bytes))
        img_array = dicom_data.pixel_array.astype(float)
        
        # Normalize the 16-bit medical data down to standard 8-bit (0-255) pixels
        if img_array.max() > 0:
            img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_array = np.uint8(img_array)
    else:
        # Load standard PNG/JPG
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img_array = np.array(img, dtype=np.uint8)
        
    # 2. DYNAMIC CLAHE
    if conversion_type == 'ct_to_mri':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    else:
        # REDUCED from 1.5 to 1.1 to prevent amplifying MRI background static
        clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8)) 
        
    img_array = clahe.apply(img_array)
    
    # 3. Downsize to 64x64 (Mandatory for your model)
    img = Image.fromarray(img_array)
    img = img.resize((64, 64), Image.LANCZOS)
    
    # 4. Normalize the image array for the AI (-1.0 to 1.0)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    # 5. Expand dimensions
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_image(prediction, conversion_type):
    # 1. Strip the batch and channel dimensions
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=-1)
    
    # 2. De-normalize back to standard pixel values (0-255)
    prediction = ((prediction + 1.0) * 127.5).astype(np.uint8)
    img = Image.fromarray(prediction, mode='L')
    
    # 3. UPSCALE THE OUTPUT TO SPECIFIC SIZE: 217x181 (Width x Height)
    img = img.resize((217, 181), Image.LANCZOS)
    
    # 4. ADVANCED DENOISING: Wipe out MRI -> CT generated static
    if conversion_type == 'mri_to_ct':
        img_array = np.array(img)
        
        # Step A: Non-Local Means Denoising (Kills speckle/static noise)
        # h=12 controls the strength. Higher = less noise, but slightly softer.
        clean_array = cv2.fastNlMeansDenoising(img_array, None, h=12, templateWindowSize=7, searchWindowSize=21)
        
        # Step B: Light Bilateral Filter (Smooths flat tissue while keeping bone boundaries sharp)
        final_array = cv2.bilateralFilter(clean_array, d=5, sigmaColor=50, sigmaSpace=50)
        
        return Image.fromarray(final_array)
    
    return img

@app.route('/convert', methods=['POST'])
def convert_image():
    if len(generators) != 2:
        return jsonify({'error': 'Models not fully loaded.'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    conversion_type = request.form.get('type', 'ct_to_mri')
    
    try:
        # Get both the bytes AND the filename so we know if it's a DICOM
        image_file = request.files['image']
        filename = image_file.filename
        image_bytes = image_file.read()
        
        # Pass the filename into the preprocessor
        input_tensor = preprocess_image(image_bytes, filename, conversion_type)
        
        # Use G for CT->MRI, and F for MRI->CT
        if conversion_type == 'ct_to_mri':
            prediction = generators['G'](input_tensor, training=False)
        else:
            prediction = generators['F'](input_tensor, training=False)
        
        # Pass conversion type to postprocessor
        output_img = postprocess_image(prediction.numpy(), conversion_type)
        
        # Save as PNG
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return jsonify({'translated_image': img_base64})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
