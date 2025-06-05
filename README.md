# 🔍 Web Logs Error Prediction Using LightGBM

This project is a Streamlit-based web application that predicts system log errors using machine learning models trained with class weighting and SMOTE for handling class imbalance.

## 🚀 Features

- Upload `.csv` log files and get real-time error predictions
- Supports both SMOTE-based and Class-Weighted LightGBM models
- Visualize error distribution and feature importance
- Download predictions as a CSV

## 🧠 Models Used

- `model_class_weight.pkl`: LightGBM with class weighting
- `model_smote.pkl`: LightGBM trained with SMOTE oversampling

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
