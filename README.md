# 🌿 Plant Disease Detection System

A deep learning-based web application that detects plant diseases from leaf images using Convolutional Neural Networks (CNN).

---

## 🚀 Features

- Upload leaf images
- Predict plant disease
- Displays disease name with confidence
- Simple and clean web interface
- Fast predictions using trained CNN model

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask (Web Framework)
- HTML, CSS (Frontend)

---

## 📁 Project Structure
plant-disease-app/
│
├── model/
│ └── plant_model.h5
│
├── static/
│ ├── css/
│ └── uploads/
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

# 1. Clone the repository
git clone https://github.com/gunatejakp/plant-disease-detection.git
cd plant-disease-detection

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py
