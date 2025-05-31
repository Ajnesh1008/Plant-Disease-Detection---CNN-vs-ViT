# Plant-Disease-Detection---CNN-vs-ViT

# 🌿 Plant Disease Detection App

An AI-powered mobile/web application that helps farmers, gardeners, and agronomists detect plant diseases in real-time using computer vision. Simply upload or capture an image of a leaf, and the app will identify the disease (if any) along with recommended solutions.

---

## 🚀 Features

- 📷 Image-based plant disease classification
- 🌱 Supports multiple plant species (e.g., tomato, potato, maize, etc.)
- 🧠 Powered by a deep learning model (CNN/MobileNetV2/etc.)
- 💡 Disease descriptions and treatment suggestions
- 🖥️ Web interface / 📱 Mobile-friendly (React Native/Flutter/etc.)
- 📊 Accuracy and confidence score for predictions
- 🌐 Offline capability (optional for mobile apps)

---

## 🧠 Tech Stack

- Frontend: React / React Native / Flutter
- Backend: Python Flask / FastAPI / Node.js
- Model: TensorFlow / PyTorch
- Deployment: Docker, Heroku / AWS / Azure
- Data: PlantVillage dataset (or custom-labeled dataset)

---

## 📸 Sample Screenshots

(Add screenshots or demo GIFs here)

---

## 🛠 Installation

### Clone the repo

    git clone https://github.com/yourusername/plant-disease-detection-app.git
    cd plant-disease-detection-app

### Backend Setup

    cd backend
    pip install -r requirements.txt
    python app.py

### Frontend Setup

    cd frontend
    npm install
    npm start

---

## 🔍 How It Works

1. User uploads or captures an image of a plant leaf.
2. Image is preprocessed and sent to the AI model.
3. Model returns the disease name and probability score.
4. App displays disease information and treatment options.

---

## 📦 Model Training (Optional)

    cd model
    python train.py --dataset ./data --epochs 30 --model plant_model.h5

(You can also use our pre-trained model available online.)

---

## 📚 Dataset

- Source: PlantVillage (open-source)
- Classes: Healthy, Early Blight, Late Blight, Leaf Mold, etc.
- Kaggle Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease

---

## 🤖 AI Model

- Architecture: MobileNetV2 (fine-tuned)
- Accuracy: ~95% on test set
- Framework: TensorFlow / Keras

---

## 🛡 Disclaimer

This app is for informational purposes only. Always consult with agricultural experts before applying treatments. The model may not generalize perfectly in all field conditions.

---

## 🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

    git checkout -b feature/your-feature
    git commit -m 'Add your feature'
    git push origin feature/your-feature

---

## 📧 Contact

Ajnesh Kumar – ajneshkumar48@gmail.com  
LinkedIn – https://linkedin.com/in/ajnesh-kumar-45b868212

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
