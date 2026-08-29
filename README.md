# Explainable Real-Time Fraud Detection Dashboard

## Overview
This dashboard provides real-time, explainable fraud detection for online payment transactions using **XGBoost** with **SHAP** explanations. Built as part of the MSc Data Science and Artificial Intelligence programme at Leeds Trinity University, it transforms a black-box fraud prediction into a transparent, business-ready tool.

## Key Features
- **Real-Time Prediction:** Sub-2 second latency for transaction screening
- **Explainability:** SHAP force plots with narrative explanations for each prediction
- **Imbalance Handling:** Class weighting (`scale_pos_weight=10.26`) to address 10:1 imbalance
- **Model:** XGBoost with ROC-AUC 0.7373 and optimised F1-Score 0.3106
- **Interactive UI:** User-friendly interface designed for fraud analysts

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
Model Performance
Metric	Score
ROC-AUC	0.7373
F1-Score	0.3106
Recall	0.4459
Precision	0.2383
Top Influential Features (SHAP Analysis)
Previous_Fraudulent_Transactions - Users with prior fraud history pose higher risk

Transaction_Amount - Unusual amounts above 95th percentile

Time_of_Transaction - Late-night transactions (22:00-06:00)

Technologies
Framework: Streamlit

Model: XGBoost (eXtreme Gradient Boosting)

Explainability: SHAP (SHapley Additive exPlanations)

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

Repository Structure
text
Fraud-Detection-App/
├── app.py                      # Streamlit dashboard
├── best_model.pkl              # Trained XGBoost model
├── best_threshold.txt          # Optimal threshold (0.59)
├── encoders.pkl                # LabelEncoders for categorical features
├── feature_names.pkl           # Feature names list
├── scaler.pkl                  # StandardScaler for normalisation
├── requirements.txt            # Python dependencies
├── Fraud_Dec_Env.ipynb         # Complete development code
└── README.md                   # Project documentation
Live Demo
  Coming Soon! This dashboard will be deployed on Streamlit Cloud.

Project Details
Institution: Leeds Trinity University

Programme: MSc Data Science and Artificial Intelligence

Supervisor: Nick Mitchell

Student: Syed Alizar Bukhari

Year: 2026

© 2026 Syed Alizar Bukhari | Leeds Trinity University
