# 🤖 AI SaaS Customer Churn Prediction

Predict customer churn using Machine Learning by analyzing user engagement, subscription behavior, and account activity. This project builds an end-to-end ML pipeline that preprocesses data, engineers meaningful features, trains multiple classification models, evaluates performance, and generates churn risk scores for proactive customer retention.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for SaaS businesses. Losing existing customers directly impacts recurring revenue and business growth.

This project predicts which customers are likely to churn by analyzing historical user activity and subscription data. It compares multiple machine learning algorithms and automatically selects the best-performing model based on evaluation metrics.

---

## 🎯 Objectives

* Predict customer churn using Machine Learning.
* Engineer meaningful customer behavior features.
* Compare multiple classification algorithms.
* Generate churn probability scores for every customer.
* Save the best trained model for future predictions.
* Create outputs that can be integrated into BI dashboards.

---

## 🚀 Features

* End-to-end Machine Learning pipeline
* Automated feature engineering
* Data preprocessing using Scikit-learn Pipelines
* Numerical feature scaling
* Categorical feature encoding
* Multiple model training and comparison
* Model evaluation using industry-standard metrics
* Feature importance analysis
* Customer churn probability prediction
* Model serialization for deployment

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Pickle
* Argparse

---

## 📂 Dataset

The model is trained using SaaS customer information such as:

* User ID
* Signup Date
* Last Login Date
* Churn Date
* Monthly Spend
* AI Features Used
* API Calls Per Month
* Support Tickets
* Subscription Plan
* Industry
* Company Size
* Country
* Churn Status

---

## ⚙️ Feature Engineering

Several new features are created to improve prediction performance:

* Customer Tenure (Days)
* Days Since Last Login
* API Calls Per Dollar Spent
* AI Feature Usage Ratio

These engineered features provide stronger indicators of customer engagement and churn risk.

---

## 🤖 Machine Learning Models

The project trains and evaluates:

* Logistic Regression
* Random Forest Classifier

The best-performing model is automatically selected using ROC-AUC score.

---

## 📊 Model Evaluation

Performance is measured using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Classification Report
* Confusion Matrix

These metrics provide a comprehensive assessment of model performance.

---

## 📈 Output Files

After execution, the project generates:

* **churn_model.pkl** — Saved trained model
* **churn_risk_scores.csv** — Customer churn probabilities
* **feature_importance.csv** — Important features influencing churn prediction

---

## 📋 Project Workflow

```text
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Data Preprocessing
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
(Logistic Regression & Random Forest)
      │
      ▼
Performance Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Generate Churn Predictions
      │
      ▼
Save Model & Reports
```

---

## 📁 Project Structure

```text
AI-SaaS-Customer-Churn-Prediction/
│
├── churn_prediction.py
├── ai_saas_users.csv
├── output/
│   ├── churn_model.pkl
│   ├── churn_risk_scores.csv
│   └── feature_importance.csv
└── README.md
```

---

## 💡 Business Value

This project enables organizations to:

* Identify customers likely to churn
* Improve customer retention strategies
* Reduce revenue loss
* Prioritize high-risk customers
* Support targeted marketing campaigns
* Enhance customer lifetime value (CLV)

---

## 🧠 Skills Demonstrated

* Machine Learning
* Classification Models
* Feature Engineering
* Data Preprocessing
* Predictive Analytics
* Customer Analytics
* Model Evaluation
* Scikit-learn Pipelines
* Business Intelligence
* Python Programming

---

## 🚀 Future Enhancements

* XGBoost and LightGBM models
* Hyperparameter tuning
* SHAP explainability
* Streamlit web application
* REST API deployment with FastAPI
* Docker containerization
* Cloud deployment on AWS or Azure

---

## ▶️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/AI-SaaS-Customer-Churn-Prediction.git
cd AI-SaaS-Customer-Churn-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python churn_prediction.py --data ai_saas_users.csv
```

---

## 👨‍💻 Author

**Piyush Palkatwar**

**Aspiring AI/ML Engineer | Data Analyst | Generative AI Enthusiast**

Passionate about building AI-powered solutions using Machine Learning, Data Science, Python, SQL, and Generative AI to solve real-world business problems.
