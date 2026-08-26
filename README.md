# 📈 Stock Market Classifier & Analytics Platform (`StockSenseML`)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Rithan7/stock-sense-ML)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.style=for-the-badge)](LICENSE)

An end-to-end Machine Learning web application built with **Flask**, **yfinance**, **Scikit-Learn**, and **Plotly** to predict stock market directional movement (Up vs. Down) using technical indicators, quantitative feature engineering, and interactive analytics dashboards.

🔗 **GitHub Repository**: [https://github.com/Rithan7/stock-sense-ML](https://github.com/Rithan7/stock-sense-ML)

---

## ✨ Features

- 🔍 **Real-Time Data Retrieval**: Fetches historical stock price & volume data seamlessly using `yfinance`.
- 📊 **Automated Feature Engineering**: Computes 14+ quantitative technical indicators:
  - **Moving Averages**: MA5, MA10, MA20, MA Crossover Signals
  - **Momentum & Trend**: RSI (Relative Strength Index), MACD & Signal Line, Momentum
  - **Volatility & Bands**: Bollinger Bands (Width & Relative Position), Historical Volatility, Price Range
- 🤖 **Machine Learning Classification**:
  - **Logistic Regression**: Baseline probabilistic linear model
  - **Random Forest Classifier**: Non-linear ensemble model with feature importance extraction
- 📈 **Interactive Plotly Visualizations**:
  - Candlestick price charts with moving averages & Bollinger Bands
  - RSI & MACD technical subplots
  - Interactive Confusion Matrix heatmaps & ROC Curves
  - Random Forest Feature Importance rankings
- 🔐 **User Authentication & Dashboard**:
  - Secure Registration & Login using `Flask-Bcrypt` & `Flask-Login`
  - User-specific Watchlist management
  - Saved Prediction History tracking with performance metrics (Accuracy, AUC, F1, Probability)

---

## 🛠️ Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Backend Framework** | Flask, Flask-SQLAlchemy, Flask-Login, Flask-Bcrypt |
| **Machine Learning** | Scikit-Learn (Logistic Regression, Random Forest), NumPy, SciPy |
| **Data Processing** | Pandas, yfinance |
| **Data Visualization**| Plotly (Interactive Javascript Charts) |
| **Database** | SQLite (Default via SQLAlchemy) |
| **Deployment Server** | Gunicorn |

---

## 📁 Project Structure

```text
Stock-Sense-ML/
├── app.py                # Main Flask application logic, data pipeline, & ML workflow
├── models.py             # Database schemas (User, Prediction, Watchlist)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git exclusion settings
└── templates/            # HTML Jinja2 templates
    ├── index.html        # Main dashboard & prediction workbench
    ├── search.html       # Stock search & analytics view
    ├── history.html      # User saved prediction history & watchlist
    ├── login.html        # Login page
    └── register.html     # Registration page
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.9+** installed on your system.
- `git` version control system.

### 1. Clone the Repository
```bash
git clone https://github.com/Rithan7/stock-sense-ML.git
cd stock-sense-ML
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional)* Customize `SECRET_KEY` and `DATABASE_URL` inside `.env`.

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🔬 Machine Learning Pipeline

1. **Data Ingestion**: Downloads historical OHLCV data for selected tickers.
2. **Preprocessing**: Cleans multi-index columns, handles missing values, and calculates technical indicator features.
3. **Target Generation**: Defines binary label $Y_{t} = 1$ if $\text{Close}_{t+1} > \text{Close}_{t}$, else $0$.
4. **Data Splitting & Scaling**: Performs temporal/random split and standardizes features using `StandardScaler`.
5. **Model Training & Evaluation**:
   - Evaluates performance via **Accuracy**, **AUC-ROC**, and **F1-Score**.
   - Outputs class probability $P(\text{Up})$ for directional bias prediction.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Rithan7/stock-sense-ML/issues).

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
