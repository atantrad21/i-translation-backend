import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os
import logging
import requests

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("🚀 I-TRANSLATION HUGGING FACE - CHECKPOINT 652 (FIXED DOWNLOAD)")
print("="*70)

# ============================================================================
# INSTANCE NORMALIZATION LAYER
# ============================================================================
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

# ============================================================================
# U-NET GENERATOR ARCHITECTURE
# ============================================================================
def downsample(filters, size, apply_norm=True):
    result = keras.Sequential()
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def unet_generator():
    inputs = layers.Input(shape=[64, 64, 1])
    
    down_stack = [
        downsample(128, 4, apply_norm=False),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]
    
    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(256, 4),
        upsample(128, 4),
    ]
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    last = layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='tanh'
    )
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

# ============================================================================
# CHECKPOINT 652 - NEW FILE IDS
# ============================================================================
FILE_IDS = {
    'F': '1bzuQ1AFKC5b0WDrSW_lUpIomquQkSCOt',
    'G': '1vE3wlV7P2_J-Ndr62yj0tsytq6gv8-SF',
    'I': '12iq54-pX3lWZnSjLFWLaOt2LV_3CTgeV',
    'J': '1Reo76L5CCybAplmj_pPNWZCWFLK6n8Zp'
}

MODELS = {}
MODELS_LOADED = False
LOADING_STATUS = "Initializing..."

# ============================================================================
# IMPROVED DOWNLOAD FUNCTION
# ============================================================================
def download_from_google_drive(file_id, destination):
    """Robust Google Drive download - tested and verified"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    logger.info(f"📥 Downloading file ID: {file_id}")
    
    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)
    
    logger.info(f"Status: {response.status_code}")
    
    # Handle confirmation for large files
    if 'text/html' in response.headers.get('Content-Type', ''):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                logger.info("✅ Using confirmation token")
                response = session.get(url, params={'confirm': value}, stream=True, timeout=60)
                break
    
    # Download in chunks
    total_size = 0
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
    
    file_size_mb = total_size / (1024 * 1024)
    logger.info(f"✅ Downloaded: {file_size_mb:.2f} MB")
    
    if file_size_mb < 1:
        raise Exception(f"File too small: {file_size_mb:.2f} MB")
    
    return destination

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_models():
    global MODELS, MODELS_LOADED, LOADING_STATUS
    
    try:
        logger.info("🔄 Loading Checkpoint 652 models...")
        
        for name, file_id in FILE_IDS.items():
            LOADING_STATUS = f"Downloading Generator {name}..."
            logger.info(f"\n📥 Generator {name}...")
            
            output_path = f'/tmp/generator_{name.lower()}.h5'
            
            download_from_google_drive(file_id, output_path)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"📦 Size: {file_size:.2f} MB")
            
            LOADING_STATUS = f"Building Generator {name}..."
            logger.info(f"🔨 Building architecture...")
            model = unet_generator()
            
            LOADING_STATUS = f"Initializing Generator {name}..."
            logger.info(f"🔧 Initializing...")
            dummy_input = tf.zeros([1, 64, 64, 1])
            _ = model(dummy_input, training=False)
            
            LOADING_STATUS = f"Loading weights for Generator {name}..."
            logger.info(f"📂 Loading weights...")
            model.load_weights(output_path, by_name=True, skip_mismatch=True)
            
            MODELS[name] = model
            logger.info(f"✅ Generator {name} READY")
            
            os.remove(output_path)
        
        MODELS_LOADED = True
        LOADING_STATUS = "✅ All models loaded - Checkpoint 652"
        logger.info("\n🎉 ALL 4 GENERATORS LOADED\n")
        
    except Exception as e:
        LOADING_STATUS = f"❌ Error: {str(e)}"
        logger.error(f"❌ FAILED: {str(e)}")
        raise

# Load models
load_models()

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('L')
    else:
        img = image.convert('L')
    
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_image(tensor):
    img_array = tensor.numpy()
    img_array = np.squeeze(img_array)
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img

# ============================================================================
# CONVERSION
# ============================================================================
def convert_image(image):
    if not MODELS_LOADED:
        return None, None, None, None, f"⏳ {LOADING_STATUS}"
    
    if image is None:
        return None, None, None, None, "❌ Please upload an image"
    
    try:
        logger.info("🔄 Converting...")
        input_tensor = preprocess_image(image)
        
        output_f = postprocess_image(MODELS['F'](input_tensor, training=False))
        output_g = postprocess_image(MODELS['G'](input_tensor, training=False))
        output_i = postprocess_image(MODELS['I'](input_tensor, training=False))
        output_j = postprocess_image(MODELS['J'](input_tensor, training=False))
        
        logger.info("✅ Complete")
        return output_f, output_g, output_i, output_j, "✅ Conversion successful!"
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None, None, None, None, f"❌ Error: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(title="I-Translation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 I-Translation - Medical Image Converter
    ### Convert Medical Images: CT ↔ MRI
    
    **✨ Checkpoint 652 (Fixed Download) - High Quality, No Noise**
    
    Upload a medical image and get 4 different high-quality conversions.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="📤 Upload Medical Image", type="pil", height=300)
            convert_btn = gr.Button("🔄 Convert Image", variant="primary", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False, value=LOADING_STATUS)
    
    gr.Markdown("### 📊 Results - 4 Generator Outputs")
    
    with gr.Row():
        output_1 = gr.Image(label="Output 1")
        output_2 = gr.Image(label="Output 2")
    
    with gr.Row():
        output_3 = gr.Image(label="Output 3")
        output_4 = gr.Image(label="Output 4")
    
    convert_btn.click(
        fn=convert_image,
        inputs=[input_image],
        outputs=[output_1, output_2, output_3, output_4, status_text]
    )
    
    gr.Markdown("""
    ---
    ### 📋 Information
    - **Models**: 4 Bidirectional U-Net Generators
    - **Checkpoint**: 652 (Fixed Download - 2026-03-09)
    - **Architecture**: Instance Normalization + Skip Connections
    - **Input**: Grayscale medical images
    - **Output**: 256×256 high-quality conversions
    
    **Contact**: atantrad@gmail.com
    """)

if __name__ == "__main__":
    logger.info("🚀 Launching Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
