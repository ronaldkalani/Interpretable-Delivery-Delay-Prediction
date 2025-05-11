# Interpretable AI System for Uncertainty-Aware Delivery Delay Prediction in Fleet Logistics Using SHAP and Probabilistic Modeling

## Goal
To build an interpretable machine learning model that predicts delivery delays and quantifies uncertainty, supporting proactive decision-making in fleet management.

## Intended Audience
- AI/ML Research Analysts
- Fleet and Logistics Managers
- Data Scientists in Transportation
- Business Intelligence Teams

## Strategy & Pipeline Steps

### Step 1: Mount Google Drive and Load Dataset
```python
from google.colab import drive
import pandas as pd
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/Fleet & Logistics/Delivery truck trip data.xlsx'
df = pd.read_excel(file_path)
df.head()
```

### Step 2: Data Cleaning & Feature Creation
```python
df.drop_duplicates(inplace=True)
df.ffill(inplace=True)
df['Planned_ETA'] = pd.to_datetime(df['Planned_ETA'])
df['actual_eta'] = pd.to_datetime(df['actual_eta'])
df['delay_minutes'] = (df['actual_eta'] - df['Planned_ETA']).dt.total_seconds() / 60
df['is_delayed'] = df['delay_minutes'] > 15
```

### Step 3: Model Dataset Preparation
```python
X = df[['TRANSPORTATION_DISTANCE_IN_KM', 'Minimum_kms_to_be_covered_in_a_day']]
y = df['is_delayed']

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

### Step 4: Train a Logistic Regression Model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]
```

### Step 5: Visualize Prediction Uncertainty
```python
import matplotlib.pyplot as plt
plt.hist(y_probs, bins=20, color='gold', edgecolor='black')
plt.title("Uncertainty in Delay Prediction")
plt.xlabel("Predicted Probability of Delay")
plt.ylabel("Number of Trips")
plt.show()
```

### Step 6: Interpret with SHAP for Transparency
```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

## Challenges
- Highly imbalanced predictions near 0.65 indicate model uncertainty
- Limited feature diversity affects performance generalization
- Requires integration of external signals (weather, traffic, driver behavior)

## Problem Statement
Fleet managers lack an explainable system to anticipate delivery delays and assess confidence in AI decisions. This project addresses that gap using delay prediction with uncertainty modeling and SHAP-based transparency.

## Dataset
**Delivery Truck Trip Data**
- Origin, Destination, Distance
- Planned vs Actual ETA
- Vehicle and Driver Info
- Delay Flag, Performance Benchmarks

## Machine Learning Prediction & Outcomes
**Histogram Insight**: Predictions cluster near 0.65, revealing uncertainty. Few predictions are near 0.25 or 0.85, where confidence is high.

**SHAP Insight**:
- Transportation Distance: Slight delay impact.
- Minimum KM/Day: Stronger predictor of delay.

## Trailer Documentation
This project illustrates how uncertainty-aware AI improves logistics by supporting better route decisions and resource planning.

## Conceptual Enhancement – AGI
Future versions could incorporate dynamic data (e.g., weather, IoT sensors) for adaptive, self-improving delivery routing.

## References
- Lundberg & Lee (2017), SHAP: https://arxiv.org/abs/1705.07874
- Abdar et al. (2021), Uncertainty in Deep Learning: https://doi.org/10.1016/j.inffus.2021.05.008
- Hosmer et al. (2013), Applied Logistic Regression, Wiley
- DHL & IBM (2018), AI in Logistics: https://www.logistics.dhl/content/dam/dhl/global/core/documents/pdf/glo-core-ai-in-logistics.pdf
