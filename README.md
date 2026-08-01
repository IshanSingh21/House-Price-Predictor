#  House Price Predictor — ML Project

An end-to-end Machine Learning project that predicts house prices based on property features. Built with **scikit-learn**, **Flask**, and a premium dark-mode web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data/generate_data.py
```

This creates `data/housing_data.csv` with 5,000 synthetic house records.

### 3. Train the model

```bash
python model/train.py
```

This trains 4 ML models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), compares them, and saves the best one.

### 4. Start the web app

```bash
python app/server.py
```

Open **http://127.0.0.1:5000** in your browser and start predicting! 

##  Models Compared

- **Linear Regression** — Baseline model
- **Decision Tree** — Interpretable tree-based model
- **Random Forest** — Ensemble of decision trees
- **Gradient Boosting** — State-of-the-art boosting (usually wins 🏆)

##  Web Interface

- Dark mode with glassmorphism design
- Interactive sliders and pill selectors
- Animated price counter
- Feature importance visualization
- Fully responsive layout

##  Project Structure

```
house-price-predictor/
├── data/
│   ├── generate_data.py        # Dataset generator
│   └── housing_data.csv        # Generated dataset
├── model/
│   ├── train.py                # Model training & evaluation
│   └── house_price_model.pkl   # Saved model
├── app/
│   ├── server.py               # Flask API server
│   ├── templates/index.html    # Web UI
│   └── static/
│       ├── style.css           # Premium styling
│       └── script.js           # Frontend logic
├── requirements.txt
└── README.md
```

##  License

This project is for educational purposes. Feel free to use and modify.
