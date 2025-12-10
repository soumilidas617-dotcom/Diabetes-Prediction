from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load ML Model
# -----------------------------
# Use joblib (consistent with how you saved it in train_model.py)
model = joblib.load("model.pkl")

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/submit_contact", methods=["POST"])
def submit_contact():
    name = request.form["name"]
    email = request.form["email"]
    subject = request.form["subject"]
    message = request.form["message"]

    print("\n--- New Contact Form Submission ---")
    print("Name :", name)
    print("Email :", email)
    print("Subject :", subject)
    print("Message :", message)
    print("-----------------------------------\n")

    return "<h2>Thank you! Your message has been submitted successfully.</h2>"


@app.route("/result", methods=["GET"])
def result():

    try:
        preg = float(request.args.get("preg"))
        glucose = float(request.args.get("glucose"))
        bp = float(request.args.get("bp"))
        skin = float(request.args.get("skin"))
        insulin = float(request.args.get("insulin"))
        bmi = float(request.args.get("bmi"))
        dpf = float(request.args.get("dpf"))
        age = float(request.args.get("age"))

        values = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        risk_score = model.predict_proba(values)[0][1] * 100

        return render_template("result.html", risk_score=round(risk_score, 2))

    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
