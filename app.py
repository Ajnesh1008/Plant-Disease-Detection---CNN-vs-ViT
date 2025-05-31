# Placeholder for Flask app
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from cnn_model import CNNModel
from vit_model import ViTModel

from torchvision import datasets

# Load class names using same dataset path used during training
#dummy_data = datasets.ImageFolder("path_to_training_data")  # Use the same path
dummy_data = datasets.ImageFolder('data/plant_disease_dataset/train')
CLASS_NAMES = list(dummy_data.classes)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
cnn_model = CNNModel()
vit_model = ViTModel()
cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=torch.device('cpu')))
vit_model.load_state_dict(torch.load('models/vit_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()
vit_model.eval()

# Define class names
#CLASS_NAMES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight','Tomato__Target_Spot','Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        cnn_output = cnn_model(img_tensor)
        vit_output = vit_model(img_tensor)

    cnn_pred = torch.argmax(cnn_output, dim=1).item()
    vit_pred = torch.argmax(vit_output, dim=1).item()

    result = {
        'cnn_prediction': CLASS_NAMES[cnn_pred],
        'vit_prediction': CLASS_NAMES[vit_pred],
        'match': CLASS_NAMES[cnn_pred] == CLASS_NAMES[vit_pred]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
