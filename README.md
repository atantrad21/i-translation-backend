---
title: I-Translation Medical Image Converter
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: mit
---

# I-Translation - Medical Image Converter

Convert medical images (CT ↔ MRI) using 4 bidirectional U-Net generators trained on Checkpoint 652.

## Features
- 4 independent U-Net generators
- Instance Normalization architecture
- High-quality conversions
- No noise or artifacts
- Checkpoint 652 (2026-03-09)

## Usage
1. Upload a medical image (CT or MRI)
2. Click "Convert Image"
3. Get 4 different high-quality outputs

## Technical Details
- **Architecture**: U-Net with Instance Normalization
- **Input**: 64×64 grayscale
- **Output**: 256×256 grayscale
- **Models**: 4 generators (F, G, I, J)
- **Total Size**: ~200 MB

## Contact
atantrad@gmail.com
