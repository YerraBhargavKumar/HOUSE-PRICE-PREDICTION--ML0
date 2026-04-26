from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])

    prediction = model.predict([[area, bedrooms, bathrooms]])[0]

    return render_template("index.html", result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)