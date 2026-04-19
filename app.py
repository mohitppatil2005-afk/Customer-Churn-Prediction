from flask import Flask, render_template, request
import pandas as pd
import pickle

# ---------- LOAD FILES ----------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

app = Flask(__name__)

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():

    # ---------- NUMERIC ----------
    tenure = float(request.form.get("tenure", 0))
    monthly = float(request.form.get("MonthlyCharges", 0))
    total = float(request.form.get("TotalCharges", 0))

    # ---------- CREATE BASE ----------
    input_dict = {col: 0 for col in columns}

    # Fill numeric
    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly
    input_dict["TotalCharges"] = total

    # ---------- BINARY ----------
    binary_fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]

    for field in binary_fields:
        input_dict[field] = int(request.form.get(field, 0))

    # ---------- SERVICES ----------
    service_fields = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    for field in service_fields:
        input_dict[f"{field}_Yes"] = int(request.form.get(field, 0))

    # ---------- CONTRACT ----------
    contract = request.form.get("Contract")
    if contract == "One year":
        input_dict["Contract_One year"] = 1
    elif contract == "Two year":
        input_dict["Contract_Two year"] = 1
    # Month-to-month → base (0)

    # ---------- INTERNET ----------
    internet = request.form.get("InternetService")
    if internet == "Fiber optic":
        input_dict["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_dict["InternetService_No"] = 1
    # DSL → base (0)

    # ---------- PAYMENT ----------
    payment = request.form.get("PaymentMethod")
    if payment == "Electronic check":
        input_dict["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        input_dict["PaymentMethod_Mailed check"] = 1
    elif payment == "Credit card (automatic)":
        input_dict["PaymentMethod_Credit card (automatic)"] = 1
    # Bank transfer → base (0)

    # ---------- DATAFRAME ----------
    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[columns]

    # ---------- SCALING ----------
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # ---------- PREDICTION ----------
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ---------- RESULT ----------
    if prob > 0.7:
        result = "High risk of churn"
    elif prob > 0.5:
        result = "Moderate risk"
    else:
        result = "Low risk"

    return render_template("result.html", result=result, prob=round(prob * 100, 2))


# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)