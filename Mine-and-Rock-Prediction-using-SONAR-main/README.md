# Mine and Rock Prediction using SONAR (ML Model)

An intelligent machine learning-based system to classify SONAR signals as **Mine** or **Rock**, helping enhance the safety of **underwater submarines** by reducing manual efforts and human error. This is a full-stack project with a trained **Random Forest** model in the backend and an interactive simulation interface in the frontend.

---

## 🚀 Project Overview

This project uses a machine learning model trained on SONAR data to detect whether an underwater object is a **Mine** or a **Rock** based on the intensity of sound wave reflections (SONAR returns). The system is designed with the goal of improving **naval safety** and minimizing risks from manually interpreting SONAR signals.

- **Dataset**: 208 rows × 61 columns  
  - **Features (60)**: Numerical intensity values normalized between 0 and 1  
  - **Label (1)**: `M` for Mine and `R` for Rock
  - **Source**: `https://www.kaggle.com/datasets/rupakroy/sonarcsv`

- **ML Model**: Random Forest Classifier  
- **Backend**: Flask API to serve the trained ML model  
- **Frontend**: React.js simulation with visual + audio alerts

---
## 🧠 Features

### ✅ Manual Prediction  
- Users can input 60 feature values to check whether the object is a Mine or Rock.

### ⚙️ Simulation Mode  
- A sequence of predictions is run iteratively.
- If a **Rock** is detected: the simulation continues normally.
- If a **Mine** is detected: a **beep sound warning** is triggered until acknowledged by the user.

---
## 🛠️ Tech Stack

- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`
- **Model**: Random Forest Classifier
- **Backend**: Flask (Python)
- **Frontend**: React.js (JavaScript), JavaScript, Tailwindcss
- **Dataset**: 208 rows, 61 columns(60 features , 1 label)
---
## 🔧 Installation
1. Clone the repo: `git clone https://github.com/ayanbeliever/Mine-and-Rock-Prediction-using-SONAR.git`
3. **Backend Setup**:
   - Navigate: `cd backend`
   - Create a virtual environment: `python -m venv .venv`
   - Activate: `source venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
   - `pip install -r requirements.txt`
   - `python app.py`
5. **Frontend Setup**:
   - `cd frontend`
   - Create a react app using vite and intall tailwindcss
   - `npm install`
   - `npm start`
---
### Manual Prediction
![image](https://github.com/user-attachments/assets/bc85369c-0d0f-4257-a4b8-97758c7523b7)
### Simulation : Rock Predicted
![image](https://github.com/user-attachments/assets/c289c704-ddeb-469b-912f-fa4a9d48b62a)
### Simulation : Mine Predicted (with beep sound until acknowledged)
![image](https://github.com/user-attachments/assets/511065bb-da08-453b-a4c1-b0cd8a6e937e)
