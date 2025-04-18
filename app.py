import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Model parameters
num_classes = None  # Will be set when model is loaded
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations - use the same as in training for consistency
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Load the model
def load_model(model_path):
    global num_classes
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine num_classes from state_dict
    # Look for 'fc.weight' to determine output shape
    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"Detected {num_classes} classes from model weights")
    else:
        raise ValueError("Could not determine number of classes from model")
    
    # Initialize model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

# Preprocess image
def preprocess_image(image):
    """Preprocess the image similar to training"""
    # Handle transparency: Replace with white background
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the original image onto the background using the alpha channel as mask
        background.paste(image, (0, 0), image.split()[3])  # 3 is the index of the alpha channel
        image = background
    else:
        # Ensure image is RGB even if not RGBA initially (e.g., grayscale)
        image = image.convert('RGB')
    
    return transform(image)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # If user does not select file, browser may submit an empty file
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        try:
            # Open and preprocess the image
            image = Image.open(file.stream)
            image_tensor = preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Define label names for each class
            label_names = [
                "Front Left Door",
                "Front Right Door",
                "Rear Left Door",
                "Rear Right Door",
                "Hood"
            ]
            
            # Format results
            results = []
            for i, prob in enumerate(probabilities):
                # Make sure we don't go out of bounds with our label array
                label = label_names[i] if i < len(label_names) else f"Class {i}"
                status = "Open" if prob > 0.5 else "Closed"
                
                results.append({
                    'class': i,
                    'label': label,
                    'status': status,
                    'probability': float(prob),
                    'prediction': 1 if prob > 0.5 else 0
                })
            
            # Create binary string representation
            binary_result = ''.join(['1' if prob > 0.5 else '0' for prob in probabilities])
            
            return jsonify({
                'binary_result': binary_result,
                'detailed_results': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Load model before starting the app
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'final_model.pth')
    model = load_model(model_path)
    print(f"Model loaded successfully. Running on {device}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
