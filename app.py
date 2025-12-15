from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "ML API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    area = data["area"]
    prediction = model.predict([[area]])
    return jsonify({"price": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
