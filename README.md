<div align="center">

# 🐘 Elephant Species Classification

### An AI-Powered Web App to Classify African vs Asian Elephants using MobileNetV2 & Flask

[![Python](https://img.shields.io/badge/Python-3.11.3-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)](#)
[![License](https://img.shields.io/badge/License-Educational%2FPortfolio-lightgrey.svg)](#)

**Upload an elephant image → Get an instant African or Asian prediction → See the confidence score & explanation.**

[🚀 Live Demo](https://elephant-species-classification.onrender.com/) · [💻 GitHub Repo](https://github.com/Mohit01112/elephant-species-classification) · [🐛 Report Bug](#) · [✨ Request Feature](#)

</div>

---

## 📖 Overview

An AI-powered web application that classifies elephant images into **African** or **Asian** elephants using **MobileNetV2**, **TensorFlow/Keras**, and **Flask**.

The project uses transfer learning with a pretrained MobileNetV2 backbone and a custom classification head. Users can upload an elephant image through the web interface and receive a predicted class, confidence score, and a simple explanation of the prediction.

---

## 🚀 Demo

🌐 **Live Application:**
https://elephant-species-classification.onrender.com/

💻 **GitHub Repository:**
https://github.com/Mohit01112/elephant-species-classification

---

## ✨ Features

| Feature | Description |
|---|---|
| 🐘 Species Classification | Distinguishes African vs Asian elephants |
| 🧠 Deep Learning | Built with MobileNetV2 |
| 🔄 Transfer Learning | Uses ImageNet pretrained weights |
| 🌐 Flask Web App | Simple upload-and-predict interface |
| 📤 Image Upload | Upload directly through the browser |
| 📊 Confidence Score | Softmax probability of the prediction |
| 🔍 Prediction Explanation | Human-readable reasoning for the result |
| 📱 Responsive UI | Clean, modern web interface |
| 💾 Latest-Image Storage | Keeps only the most recent uploaded image locally |
| 🧪 Train/Test Split | Separate datasets for training and evaluation |

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|---|---|
| **Python 3.11.3** | Programming language |
| **TensorFlow / Keras** | Deep learning and inference |
| **MobileNetV2** | Image classification backbone |
| **Flask** | Backend web framework |
| **NumPy** | Numerical operations |
| **Pillow** | Image processing |
| **HTML / CSS** | Frontend |
| **Git / GitHub** | Version control |

</div>

---

## 🧠 Model Architecture

The project uses **MobileNetV2** as the base convolutional neural network.

```
Input Image
    ↓
224 × 224 × 3
    ↓
MobileNetV2
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU)
    ↓
Dropout
    ↓
Dense(2, Softmax)
    ↓
African / Asian
```

The MobileNetV2 base is initialized with ImageNet pretrained weights, and the convolutional base is frozen during inference. The final classification layer contains two output classes.

---

## 🔄 How the Application Works

```
User uploads image
        ↓
Flask receives image
        ↓
Image resized to 224 × 224
        ↓
Image converted to NumPy array
        ↓
Pixel values normalized by 255
        ↓
MobileNetV2 model
        ↓
Softmax probabilities
        ↓
Highest-probability class selected
        ↓
Result page
        ↓
Prediction + Confidence + Explanation
```

---

## 📂 Project Structure

```
elephant-species-classification/
│
├── app.py
├── requirements.txt
├── class_indices.json
├── best_mobilenetv2.weights.h5
├── elephant_model_xc.h5
├── .gitignore
│
├── train/
│   ├── african/
│   │   ├── image files...
│   │   └── ...
│   └── asian/
│       ├── image files...
│       └── ...
│
├── test/
│   ├── african/
│   │   ├── image files...
│   │   └── ...
│   └── asian/
│       ├── image files...
│       └── ...
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── uploads/
        └── .gitkeep
```

### 📌 Important Files

| File | Description |
|---|---|
| `app.py` | Flask backend responsible for loading the model, receiving uploaded images, preprocessing them, generating predictions, and rendering the result page. |
| `best_mobilenetv2.weights.h5` | Saved trained weights for the MobileNetV2-based classifier. |
| `elephant_model_xc.h5` | A previously trained model file included in the repository. |
| `class_indices.json` | Stores the mapping between class names and numeric class indices. |
| `templates/index.html` | Main upload page. |
| `templates/result.html` | Prediction result page. |
| `train/` | Training dataset. |
| `test/` | Testing dataset. |

---

## 📊 Dataset

The dataset is organized into two classes: **African** and **Asian**.

The training and testing directories follow the folder-based image classification format expected by Keras:

```
train/
├── african/
└── asian/

test/
├── african/
└── asian/
```

Each class folder contains the corresponding elephant images.

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/Mohit01112/elephant-species-classification.git
cd elephant-species-classification
```

**2. Create a virtual environment** *(Windows PowerShell)*
```powershell
python -m venv venv
```

**3. Activate the virtual environment**
```powershell
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks activation:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```
> Then activate again:
> ```powershell
> .\venv\Scripts\Activate.ps1
> ```

**4. Install dependencies**
```powershell
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```powershell
python app.py
```

The application will be available at:

```
http://127.0.0.1:5000
```

Open the address in your browser.

---

## 📸 Using the Application

1. Open the Flask web application.
2. Upload an elephant image in `.jpg`, `.jpeg`, or `.png` format.
3. Click **Analyze Elephant**.
4. The model processes the image.
5. A result page displays:
   - 🖼️ Uploaded image
   - 🎯 Predicted elephant class
   - 📊 Model confidence
   - 🔍 Explanation of the prediction
6. Click **Analyze Another Image** to make a new prediction.

Only the latest uploaded image is stored, at:

```
static/uploads/uploaded_elephant.jpg
```

A new upload replaces the previous image.

---

## 🔍 Prediction Explanation

The application provides a simple explanation based on the characteristics generally associated with the predicted class and the model's confidence.

**African elephant** predictions may be associated with features such as:
- Larger ears
- Broader body shape
- Characteristic head and trunk features

**Asian elephant** predictions may be associated with:
- Smaller, rounded ears
- More compact body shape
- Different head and trunk characteristics

> **Note:** The explanation is a human-readable interpretation of the prediction. The standard MobileNetV2 classifier itself produces class probabilities and does not directly generate natural-language reasoning. For a stronger explainability implementation, techniques such as **Grad-CAM** could be added to visualize the image regions influencing the prediction.

---

## 📈 Confidence Score

The application calculates confidence from the highest softmax probability:

```python
confidence = float(np.max(preds)) * 100
```

**Example:**
```
Prediction: African
Confidence: 94.25%
```

This confidence value is the model's predicted probability for the selected class. It should not be interpreted as guaranteed real-world accuracy for an individual image.

---

## 🧪 Training and Testing

The project contains separate datasets for `train/` and `test/`.

The training data is used to learn the classification task, while the testing data is used to evaluate model performance on unseen images.

The final Flask application does not retrain the model — it loads the previously trained weights and performs inference on uploaded images.

---

## 🔐 Files Excluded from Git

The `.gitignore` file excludes development and temporary files such as:

```
venv/
.env/
__pycache__/
.ipynb_checkpoints/
static/uploads/*
```

This keeps the repository cleaner and prevents the local virtual environment and uploaded user images from being committed.

---

## ⚠️ Model File Size

The repository contains large model files.

`elephant_model_xc.h5` is approximately **86 MB**, which is above GitHub's recommended 50 MB file size warning threshold, although the push succeeded.

For a production repository, large model files can be managed with:
- Git LFS
- Hugging Face
- Cloud / object storage
- Other model hosting solutions

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

`Image Classification` · `Convolutional Neural Networks` · `Transfer Learning` · `MobileNetV2` · `TensorFlow/Keras` · `Image Preprocessing` · `Softmax Classification` · `Model Inference` · `Flask Web Development` · `HTML/CSS Frontend Integration` · `Virtual Environments` · `Git & GitHub`

---

## 👨‍💻 Author

**Mohit**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Mohit01112-181717?logo=github&logoColor=white)](https://github.com/Mohit01112)

🌐 **[Live Demo](https://elephant-species-classification.onrender.com/)** &nbsp;|&nbsp; 💻 **[GitHub Repository](https://github.com/Mohit01112/elephant-species-classification)**

⭐ **If you found this project useful, consider giving it a star!** ⭐

</div>
