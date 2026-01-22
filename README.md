# ✈️ Flight Price Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced ML-powered flight price prediction with 94.14% accuracy using XGBoost**

![Flight Price Predictor](https://img.shields.io/badge/Accuracy-94.14%25-success)
![MAE](https://img.shields.io/badge/MAE-Rs.606-blue)
![R² Score](https://img.shields.io/badge/R²-0.9414-green)

---

## 🌟 **Features**

- 🎯 **Real-time Price Predictions** - Get instant flight price estimates
- 📊 **Interactive Dashboard** - Explore model performance with beautiful visualizations
- 📈 **Feature Analysis** - Understand which factors influence prices
- 🔍 **Model Comparison** - Compare multiple ML algorithms
- ✅ **High Accuracy** - 94.14% R² score with ±Rs.606 average error

---

## 🚀 **Live Demo**

👉 **[Try it now!](https://your-app-name.streamlit.app)**

---

## 📊 **Model Performance**

| Metric | Value |
|--------|-------|
| **Test MAE** | Rs.605.73 |
| **Test RMSE** | Rs.1,104.38 |
| **Test R²** | 0.9414 |
| **Test MAPE** | 7.03% |
| **Predictions within ±20%** | 92% |
| **Training Samples** | 8,369 |
| **Test Samples** | 2,093 |

---

## 🛠️ **Tech Stack**

### **Machine Learning**
- **XGBoost** - Gradient boosting algorithm
- **Scikit-learn** - Model evaluation and preprocessing
- **Pandas & NumPy** - Data manipulation

### **Frontend**
- **Streamlit** - Interactive web application
- **Plotly** - Dynamic visualizations
- **Custom CSS** - Modern UI design

---

## 📦 **Installation**

### **Option 1: Run Locally**

```bash
# Clone repository
git clone https://github.com/nourlouta/flight-price-prediction.git
cd flight-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py

Option 2: Docker (Optional)
# Build image
docker build -t flight-predictor .

# Run container
docker run -p 8501:8501 flight-predictor

📖 Usage

1. Predict Flight Price
Navigate to the 🎯 Predict Price tab
Enter flight details:
Airline (IndiGo, Air India, etc.)
Route (Source → Destination)
Date & Time
Number of stops
Duration
Click "Predict Price"
Get instant price estimate with confidence interval!

2. Explore Model Performance
View accuracy metrics
Compare different ML models
Analyze prediction errors
Explore feature importance