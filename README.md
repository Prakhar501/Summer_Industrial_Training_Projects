# ☀️ Solar Panel Defect Classification using EfficientNetB0

An AI-powered web application for automated **solar panel defect classification** using **EfficientNetB0** and **Streamlit**. The system classifies solar panel images into six categories, enabling faster and more reliable inspection compared to manual methods.

---

## 📌 Project Overview

Solar panels are exposed to environmental conditions that can reduce their efficiency due to dirt accumulation, bird droppings, snow, or physical/electrical damage. Manual inspection of large-scale solar farms is time-consuming and expensive.

This project leverages **Transfer Learning** with **EfficientNetB0** to automatically identify different types of defects from RGB images.

---

## 🎯 Objectives

- Automate solar panel defect detection.
- Compare multiple deep learning models.
- Select the best-performing architecture.
- Deploy the final model using Streamlit.
- Provide real-time predictions with confidence scores.

---

## 🗂 Dataset

The dataset contains images belonging to **6 classes**:

- 🐦 Bird-drop
- ✅ Clean
- 🌫 Dusty
- ⚡ Electrical Damage
- 🔨 Physical Damage
- ❄ Snow

### Dataset Split

| Dataset | Images |
|---------|-------:|
| Training | ~5100 |
| Validation | 137 |
| Testing | 137 |

> Images were augmented offline before training to improve model generalization.

---

## 🧠 Models Evaluated

- Custom CNN
- MobileNetV2
- EfficientNetB0 ✅ (Selected)
- ResNet50

---

## 📊 Model Performance

| Model | Training Accuracy | Validation Accuracy |
|--------|------------------:|--------------------:|
| CNN | 99.67% | 66.15% |
| MobileNetV2 | 92.88% | 82.31% |
| ResNet50 | 85.80% | 85.38% |
| **EfficientNetB0** | **89.00%** | **86.92%** |

**EfficientNetB0** was selected due to its superior validation accuracy and better generalization performance.

---

## 🏗 Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Streamlit
- Git & GitHub

---

## 📂 Project Structure

```
SolarPanel-Defect-Detection/
│
├── app.py
├── efficientnetb0.keras
├── requirements.txt
├── README.md
├── notebooks/
│   └── model_training.ipynb
│
└── images/
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/SolarPanel-Defect-Detection.git
```

Move into the project folder:

```bash
cd SolarPanel-Defect-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🖼 Application Workflow

1. Upload a solar panel image.
2. Image is resized to **224 × 224** pixels.
3. EfficientNetB0 performs inference.
4. Softmax probabilities are generated.
5. The predicted defect class and confidence score are displayed.

---

## 📈 Features

- Image upload interface
- Real-time defect prediction
- Confidence score
- Top-3 predictions
- Probability visualization
- Clean and responsive Streamlit UI

---

## 🚧 Challenges Faced

- CNN overfitting
- Transfer learning model selection
- Hyperparameter tuning
- Preprocessing consistency
- Model deployment on Streamlit
- GitHub model size limitations
- Kaggle model export issues

---

## ✅ Solutions Implemented

- Transfer Learning using EfficientNetB0
- Dropout layers
- L2 Regularization
- EarlyStopping
- ReduceLROnPlateau
- Offline data augmentation
- Streamlit deployment
- Model optimization for inference

---

## 📌 Future Scope

- Object detection using YOLOv8
- Drone-based automated inspection
- Thermal image analysis
- Explainable AI using Grad-CAM
- Mobile application deployment
- Edge AI implementation

---

## 👨‍💻 Author

**Prakhar Bhatnagar**

B.Tech Artificial Intelligence & Machine Learning  
Guru Gobind Singh Indraprastha University (GGSIPU)

---

## 📜 License

This project is developed for **academic and industrial training purposes**.

---
