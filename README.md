# 🍷 Wine Quality Prediction

A Machine Learning web app that predicts wine quality using **XGBoost** — trained on 6,500+ red and white wines from the UCI Wine Quality dataset.

🔗 **Live Demo:** [Click here to try the app](https://wine-quality-prediction-f4knowahxunzmfosfk48tj.streamlit.app)

---

## 📌 Project Overview

Wine quality assessment is traditionally done by human experts, which is expensive and subjective. This app automates the process using physicochemical properties of wine to predict whether a wine is **Good** or **Bad** and estimate its **quality score (3–8)**.

---

## 📁 Project Structure

```
wine-quality-prediction/
├── pkl/
│   ├── wine_model.pkl       # Trained XGBoost model
│   └── wine_scaler.pkl      # StandardScaler
├── Model/
│   └── wine_quality_prediction.ipynb  # Training notebook
├── application.py           # Streamlit app
├── requirements.txt
├── runtime.txt              # Python 3.11
└── README.md
```

---

## ✨ Features

- 🔍 **Single Prediction** — Input wine properties and get instant Good/Bad classification + quality score
- 📂 **Batch Prediction** — Upload a CSV to predict quality across multiple wines at once
- 📥 **Download Results** — Export batch predictions as CSV
- 📊 **Confidence Score** — Shows probability of being a good wine

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Classifier |
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| Target | Good Wine (quality ≥ 7) vs Bad Wine |

---

## 📊 Dataset

- **Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Red wine:** 1,599 samples
- **White wine:** 4,898 samples
- **Total:** 6,497 samples
- **Features:** 11 physicochemical properties + wine type

---

## 🚀 Run Locally

```bash
git clone https://github.com/Dhairya-45/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
streamlit run application.py
```

---

## 👨‍💻 Author

**Dhairya** — [GitHub](https://github.com/Dhairya-45)