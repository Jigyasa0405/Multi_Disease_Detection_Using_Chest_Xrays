import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import base64

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disease classes
CLASSES = ['COVID-19', 'Cardiomegaly', 'Normal', 'Pleural Effusion', 'Pneumonia', 'Tuberculosis']

# Model initialization
class XRayModel:
    def __init__(self, model_path=None):
        # Load EfficientNet-B4
        self.model = models.efficientnet_b4(pretrained=False)
        
        # Modify classifier for 6 classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 6)
        )
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: No model weights loaded. Using random initialization.")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess the input image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0).to(device)
    
    def predict(self, image_tensor):
        """Make prediction"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Hook the last convolutional layer of EfficientNet-B4
        target_layer = self.model.features[-1][0]
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor, class_idx):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        self.model.zero_grad()
        output = self.model(image_tensor)
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        feature_maps = self.feature_maps[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Weighted combination of feature maps
        cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def apply_colormap_on_image(self, original_image, cam, alpha=0.5):
        """Overlay heatmap on original image"""
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
        
        # Apply percentile thresholding
        threshold = np.percentile(cam_resized, 20)
        cam_resized[cam_resized < threshold] = 0
        
        # Apply gamma correction
        gamma = 0.7
        cam_resized = np.power(cam_resized, gamma)
        
        # Normalize again
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        img_array = np.array(original_image.resize((original_image.width, original_image.height)))
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Blend images
        superimposed = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
        
        return superimposed, heatmap

# Initialize model and Grad-CAM
xray_model = XRayModel(model_path='efficientnet_b4_best.pth')  # Update path as needed
grad_cam = GradCAM(xray_model.model)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'device': str(device)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        original_image = image.copy()
        
        # Preprocess for model
        image_tensor = xray_model.preprocess_image(image)
        
        # Make prediction
        predicted_class, confidence, all_probabilities = xray_model.predict(image_tensor)
        
        # Generate Grad-CAM visualization
        cam = grad_cam.generate_cam(image_tensor, predicted_class)
        overlay_image, heatmap = grad_cam.apply_colormap_on_image(
            original_image, cam, alpha=0.5
        )
        
        # Convert images to base64
        def image_to_base64(img_array):
            img = Image.fromarray(img_array.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        overlay_b64 = image_to_base64(overlay_image)
        heatmap_b64 = image_to_base64(heatmap)
        
        # Prepare top 3 predictions
        top_3_indices = np.argsort(all_probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                'disease': CLASSES[idx],
                'probability': float(all_probabilities[idx]),
                'percentage': f"{all_probabilities[idx] * 100:.2f}%"
            }
            for idx in top_3_indices
        ]
        
        # Prepare response
        response = {
            'prediction': {
                'disease': CLASSES[predicted_class],
                'confidence': float(confidence),
                'confidence_percentage': f"{confidence * 100:.2f}%"
            },
            'all_probabilities': {
                CLASSES[i]: {
                    'value': float(all_probabilities[i]),
                    'percentage': f"{all_probabilities[i] * 100:.2f}%"
                }
                for i in range(len(CLASSES))
            },
            'top_3_predictions': top_3_predictions,
            'visualizations': {
                'overlay': overlay_b64,
                'heatmap': heatmap_b64
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': CLASSES})

if __name__ == '__main__':
    print(f"Running on device: {device}")
    print(f"Model classes: {CLASSES}")
    app.run(host='0.0.0.0', port=5000, debug=True)
