import os
import io
import cv2
import torch
import numpy as np
import warnings
import requests
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

# Suppress all warnings
warnings.filterwarnings('ignore')

def download_model(url, filename):
    os.makedirs('pretrained_models', exist_ok=True)
    filepath = os.path.join('pretrained_models', filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists.")

# Download models
download_model(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', 
    'RealESRGAN_x2plus.pth'
)

download_model(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 
    'RealESRGAN_x4plus.pth'
)

app = Flask(__name__)

def upscale_image(image_array, scale=2):
    # Prepare model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    # Find model path 
    model_path = os.path.join(os.getcwd(), 'pretrained_models', f'RealESRGAN_x{scale}plus.pth')
    
    # Initialize upscaler
    upscaler = RealESRGANer(
        scale=scale, 
        model_path=model_path, 
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    
    # Upscale image
    output, _ = upscaler.enhance(image_array)
    return output

@app.route('/enhance_2x', methods=['POST'])
def enhance_2x():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    image_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    enhanced_array = upscale_image(image_array, scale=2)
    
    # Convert to bytes for sending
    _, buffer = cv2.imencode('.jpg', enhanced_array)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/enhance_4x', methods=['POST'])
def enhance_4x():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    image_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    enhanced_array = upscale_image(image_array, scale=4)
    
    # Convert to bytes for sending
    _, buffer = cv2.imencode('.jpg', enhanced_array)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)