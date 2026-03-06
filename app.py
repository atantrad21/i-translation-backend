import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
import zipfile
import io

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

    # UPDATED: Match the down_stack from the training notebook (6 layers)
    down_stack = [
        downsample(128, 4, False, name=f'{name}_down1'), # (bs, 32, 32, 128), apply_norm=False
        downsample(256, 4, name=f'{name}_down2'), # (bs, 16, 16, 256)
        downsample(256, 4, name=f'{name}_down3'), # (bs, 8, 8, 256)
        downsample(256, 4, name=f'{name}_down4'), # (bs, 4, 4, 256)
        downsample(256, 4, name=f'{name}_down5'), # (bs, 2, 2, 256)
        downsample(256, 4, name=f'{name}_down6')  # (bs, 1, 1, 256)
    ]

    # UPDATED: Match the up_stack from the training notebook (5 layers)
    up_stack = [
        upsample(256, 4, True, name=f'{name}_up1'),  # (bs, 2, 2, 256), apply_dropout=True
        upsample(256, 4, True, name=f'{name}_up2'),  # (bs, 4, 4, 256), apply_dropout=True
        upsample(256, 4, name=f'{name}_up3'),        # (bs, 8, 8, 256)
        upsample(256, 4, name=f'{name}_up4'),        # (bs, 16, 16, 256)
        upsample(128, 4, name=f'{name}_up5')         # (bs, 32, 32, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, activation='tanh',
                                 name=f'{name}_output') # (bs, 64, 64, 1)

    concat = layers.Concatenate()

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    return keras.Model(inputs=inputs, outputs=x, name=name)

print("\n" + "="*80)
print("I-TRANSLATION: CT ↔ MRI CONVERSION (Checkpoint 652)")
print("="*80 + "\n")

# Find weight files
print("Searching for weight files...")
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
print(f"Files in directory: {os.listdir(current_dir)}")

GENERATORS = {}
FILE_NAMES = {
    'f': 'generator_f.h5',
    'g': 'generator_g.h5',
    'i': 'generator_i.h5',
    'j': 'generator_j.h5'
}

for gen_name, file_name in FILE_NAMES.items():
    file_path = file_name  # Gradio uses current directory
    print(f"\n[{gen_name.upper()}] Looking for {file_path}...")

    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[{gen_name.upper()}] ✓ Found! Size: {file_size:.2f} MB")

        try:
            print(f"[{gen_name.upper()}] Building architecture...")
            model = unet_generator(name=f'generator_{gen_name}')

            print(f"[{gen_name.upper()}] Initializing layers...")
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)

            print(f"[{gen_name.upper()}] Loading weights...")
            model.load_weights(file_path)

            print(f"[{gen_name.upper()}] ✓ SUCCESS!")
            GENERATORS[gen_name] = model

        except Exception as e:
            print(f"[{gen_name.upper()}] ✘ FAILED: {str(e)}")
    else:
        print(f"[{gen_name.upper()}] ✘ NOT FOUND")

print("\n" + "="*80)
print(f"MODELS LOADED: {len(GENERATORS)}/4")
print("="*80 + "\n")

if len(GENERATORS) == 4:
    print("✓ ALL MODELS READY!")
else:
    print("✘ ERROR: Missing weight files!")
    print("Please upload all 4 .h5 files to Space root")

def convert_image(input_image):
    if len(GENERATORS) != 4:
        return None, None, None, None, "❌ Models not loaded! Upload .h5 files to Space root."

    if input_image is None:
        return None, None, None, None, "❌ Please upload an image"

    try:
        # Preprocess
        img = Image.fromarray(input_image).convert('L')
        img = img.resize((64, 64), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = img_array[np.newaxis, :, :, np.newaxis]
        input_tensor = tf.constant(img_array)

        # Convert with all 4 generators
        outputs = []
        for gen_name in ['f', 'g', 'i', 'j']:
            output_tensor = GENERATORS[gen_name](input_tensor, training=False)
            output_array = output_tensor.numpy()[0]
            output_array = ((output_array + 1.0) * 127.5).astype(np.uint8)
            output_array = output_array[:, :, 0]
            output_img = Image.fromarray(output_array, mode='L')
            output_img = output_img.resize((256, 256), Image.LANCZOS)
            outputs.append(np.array(output_img))

        status = f"✅ Conversion successful! Generated 4 outputs from checkpoint 652"
        return outputs[0], outputs[1], outputs[2], outputs[3], status

    except Exception as e:
        return None, None, None, None, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="I-Translation: CT ↔ MRI") as demo:
    gr.Markdown("""
    # ðŸ˜‡ I-Translation: Medical Image Translation
    ## CT ↔ MRI Conversion using CycleGAN (Checkpoint 652)

    Upload a medical image and get 4 different translation results from independent generators.
    """)

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="ðŸ˜Š Input Image (CT or MRI)", type="numpy")
            convert_btn = gr.Button("↺ Convert", variant="primary", size="lg")

        with gr.Column():
            output1 = gr.Image(label="Generator F Output", type="numpy")
            output2 = gr.Image(label="Generator G Output", type="numpy")

    with gr.Row():
        output3 = gr.Image(label="Generator I Output", type="numpy")
        output4 = gr.Image(label="Generator J Output", type="numpy")

    status = gr.Textbox(label="Status", interactive=False)

    convert_btn.click(
        fn=convert_image,
        inputs=[input_img],
        outputs=[output1, output2, output3, output4, status]
    )

    gr.Markdown(f"""
    ### ℹ️ Model Status
    - **Models Loaded:** {len(GENERATORS)}/4
    - **Checkpoint:** 652
    - **Version:** 6.0.0 (Gradio)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
