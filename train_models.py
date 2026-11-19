import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ==========================
# LOAD DATA
# ==========================

train = pd.read_csv("dataset/weather_train.csv")
test = pd.read_csv("dataset/weather_test.csv")

df = pd.concat([train, test], ignore_index=True)
print("Dataset shape:", df.shape)

df = df.dropna()

# ==========================
# CREATE RAIN CATEGORY
# ==========================

def categorize_rain(x):
    if x <= 2.5:
        return "Low"
    elif x <= 7.5:
        return "Moderate"
    elif x <= 35:
        return "High"
    else:
        return "Extreme"

df["RainCategory"] = df["Rainfall"].apply(categorize_rain)

# ==========================
# SELECT FEATURES (8 INPUTS)
# ==========================

features = [
    "Humidity3pm",
    "Pressure3pm",
    "Cloud3pm",
    "Temp3pm",
    "WindGustSpeed",
    "Humidity9am",
    "Pressure9am",
    "Cloud9am"
]

X = df[features]
y_reg = df["Rainfall"]
y_class = df["RainCategory"]

# ==========================
# SCALE
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

_, _, y_class_train, y_class_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# ==========================
# REGRESSION
# ==========================

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train, y_reg_train)

reg_preds = rf_reg.predict(X_test)

print("\nRegression Results")
print("RMSE:", np.sqrt(mean_squared_error(y_reg_test, reg_preds)))
print("R2:", r2_score(y_reg_test, reg_preds))

# ==========================
# CLASSIFICATION
# ==========================

rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_class_train)

class_preds = rf_clf.predict(X_test)

print("\nClassification Accuracy:",
      accuracy_score(y_class_test, class_preds))

# ==========================
# SAVE MODELS
# ==========================

os.makedirs("models", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

joblib.dump(rf_reg, "models/rf_reg.pkl")
joblib.dump(rf_clf, "models/rf_clf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/feature_names.pkl")

print("Models saved!")

# ==========================
# VISUALIZATIONS
# ==========================

# Rainfall Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Rainfall"], bins=40, kde=True)
plt.title("Rainfall Distribution")
plt.savefig("static/graphs/rainfall_distribution.png")
plt.close()

# Correlation Heatmap (Numeric Only)
plt.figure(figsize=(10,7))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("static/graphs/correlation_heatmap.png")
plt.close()

# Feature Importance
importances = rf_reg.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.savefig("static/graphs/feature_importance.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_class_test, class_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Classification Confusion Matrix")
plt.savefig("static/graphs/confusion_matrix.png")
plt.close()

print("Graphs saved successfully!")