# 🧠 AI-Powered Anomaly Detection in Tourism Data  
## Early Identification of Destination Overcrowding

This repository contains an end-to-end machine learning workflow for **detecting anomalies in tourism datasets** to identify early signs of **destination overcrowding**. The project explores both baseline ML models and an advanced hybrid model combining traditional and deep learning techniques to improve detection accuracy.

---

## 🚀 Features
- ✔️ Data preprocessing and feature engineering  
- ✔️ Baseline model using **CatBoost**  
- ✔️ Proposed hybrid model **TH-StackNet**  
- ✔️ Model comparison & evaluation  
- ✔️ Explainability with **SHAP analysis**  
- ✔️ Complete Jupyter Notebook included  

---

## 📁 Project Structure

├── AI_Powered_Anomaly_Detection_in_Tourism_Data.ipynb
├── README.md
└── data/ (optional – add your dataset here)


---

## 🏗️ Model Architecture

This project uses two main modeling approaches:

---

# **1️⃣ Baseline Model: CatBoost Classifier**

A powerful gradient-boosting algorithm designed for:
- Handling categorical + numerical data automatically  
- Fast training  
- High accuracy on tabular datasets  
- Built-in regularization to reduce overfitting  

**CatBoost Workflow:**
Input Data → Encoding/Preprocessing → CatBoost Training → Prediction → Anomaly Score


---

# **2️⃣ Proposed Hybrid Model: TH-StackNet**

The proposed model **TH-StackNet** (Tourism Hybrid Stacked Network) combines  
traditional ML algorithms with a neural network to maximize anomaly detection accuracy.

The architecture follows a **stacked ensemble** design:

             ┌───────────────────────────┐
             │     Input Feature Set      │
             └──────────────┬────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
  ┌─────────────┐   ┌─────────────┐    ┌────────────────┐
  │ ML Model 1   │   │ ML Model 2   │    │ Neural Network │
  │ (e.g., RF)   │   │ (e.g., XGB)  │    │   (Dense NN)   │
  └─────────────┘   └─────────────┘    └────────────────┘
  
          │                 │                  │
          └───────────┬─────┴──────┬──────────┘
                      ▼             ▼
               ┌────────────────────────┐
               │   Stacking Layer        │
               │  (Meta-Learner Model)   │
               └──────────────┬─────────┘
                              ▼
                     ┌────────────────┐
                     │  Final Output  │
                     │ (Anomaly Flag) │
                     └────────────────┘
                     

---

## 🔍 TH-StackNet Components

### **🔹 Level-1 Models (Base Learners)**
- Random Forest  
- XGBoost  
- Neural Network (Fully connected layers)

These models learn independently and extract different feature relationships.

### **🔹 Level-2 Model (Meta-Learner)**
A lightweight ML model (e.g., Logistic Regression or LightGBM) that:
- Takes predictions from Level-1 models  
- Learns optimal combination weights  
- Produces the final anomaly classification  

---

## ⚙️ Neural Network Sub-Architecture
Input Layer
↓
Dense (64 units, ReLU)
↓
Dropout (0.3)
↓
Dense (32 units, ReLU)
↓
Dropout (0.2)
↓
Output Layer (Sigmoid)



---

## 🎯 Why TH-StackNet Works Better
- Ensemble reduces variance and bias  
- Neural network captures non-linear patterns  
- ML models capture tree-based interactions  
- Meta-learner blends strengths of all models  
- More stable and robust for anomaly detection

---

## 📌 Summary Table

| Component | Type | Purpose |
|----------|------|---------|
| CatBoost | Baseline | Benchmark model |
| RF + XGB | Base Learners | Tree-based feature interactions |
| Neural Network | Base Learner | Non-linear pattern extraction |
| Meta-Learner | Final Layer | Combines all model outputs |

---


---

## 📊 Explainability (SHAP)
The notebook includes **SHAP value analysis** to understand:
- Key feature contributions  
- How model decisions vary  
- Global vs. local interpretability  

---

