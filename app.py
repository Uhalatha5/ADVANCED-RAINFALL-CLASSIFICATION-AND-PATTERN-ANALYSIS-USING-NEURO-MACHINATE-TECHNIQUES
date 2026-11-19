from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
rf_reg = joblib.load("models/rf_reg.pkl")
rf_clf = joblib.load("models/rf_clf.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/feature_names.pkl")

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    values = []

    for feature in features:
        values.append(float(request.form[feature]))

    input_array = np.array([values])
    input_scaled = scaler.transform(input_array)

    rainfall = rf_reg.predict(input_scaled)[0]
    category = rf_clf.predict(input_scaled)[0]

    return render_template("index.html",
                           features=features,
                           rainfall=round(rainfall, 2),
                           category=category)

@app.route("/charts")
def charts():
    return render_template("charts.html")

if __name__ == "__main__":
    app.run(debug=True)