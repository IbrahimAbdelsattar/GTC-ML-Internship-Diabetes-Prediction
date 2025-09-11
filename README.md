# 🩺 Diabetes Prediction Project  

## 📌 Overview  
This project is part of the **ML Internship at GTC** and focuses on predicting whether a patient is **Diabetic** or **Non-Diabetic** based on health-related metrics. Early detection of diabetes is crucial because it allows individuals and healthcare professionals to take preventive measures and manage the disease effectively before it leads to severe complications.  

By leveraging **Machine Learning (ML)** models and a clean pipeline, this project provides a simple yet powerful tool that can assist in healthcare decision-making.  

---

## 🎯 Why This Project is Important  
- **Healthcare impact**: Diabetes is one of the most common chronic diseases worldwide, and early prediction helps reduce risks.  
- **Data-driven decisions**: Using machine learning to identify patterns in health metrics makes the prediction process objective and scalable.  
- **Accessibility**: A deployed app (via Streamlit) makes predictions easy for non-technical users (patients or doctors).  

---

## 📊 Dataset  
We used the **Pima Indians Diabetes Dataset**, which contains health records with the following features:  

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI (Body Mass Index)  
- Diabetes Pedigree Function (DPF)  
- Age  

Target variable:  
- **Outcome** → `0 = Non-Diabetic`, `1 = Diabetic`  

---

## ⚙️ Project Steps  

### 1️⃣ Data Exploration & Preprocessing  
- Handled missing values.  
- Conducted **Exploratory Data Analysis (EDA)** with visualizations like histograms, scatterplots, and correlation heatmaps.  
- Normalized the data for consistent scaling across features.  

### 2️⃣ Model Training  
We applied multiple machine learning models and tuned their hyperparameters using **GridSearchCV**:  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  

Each model was compared based on **cross-validation accuracy**.  

### 3️⃣ Handling Imbalanced Data  
Instead of using **SMOTE** (which caused version issues), we applied **class weights** in models like Logistic Regression, SVM, and Random Forest to balance diabetic vs non-diabetic cases.  

### 4️⃣ Deep Learning Model (DNN)  
We built a **Deep Neural Network (DNN)** using TensorFlow/Keras to further test if deep learning could improve accuracy.  
- Architecture: Multiple dense layers with ReLU activation and dropout for regularization.  
- Result: Comparable accuracy with classical ML models, but overfitting required tuning.  

### 5️⃣ Best Model Selection  
After tuning, the best-performing models were:  
- **Logistic Regression** → Best CV Accuracy: ~77.85%  
- **Support Vector Machine** → Best CV Accuracy: ~77.85%  
- Random Forest and Gradient Boosting followed closely.  

We saved the best models using **joblib** for deployment.  

### 6️⃣ Deployment with Streamlit  
We created an interactive **Streamlit app** where users can input patient data and instantly get predictions.  
- Input: User enters features like glucose level, BMI, insulin, etc.  
- Output: Model predicts whether the patient is diabetic or not, with probability score.  

---

## 🚀 How to Use the Project  

### 🔧 Installation  
Clone this repository and install the required dependencies:  
```bash
pip install -r requirements.txt

👥 Who Can Use This Project

-Healthcare professionals: As a quick screening tool for early detection.
-Patients: To self-monitor risk based on health metrics.
-Researchers & Students: As a reference for classification ML pipelines and healthcare AI projects.
