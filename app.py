from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    vec = vectorizer.transform([data])
    prediction = model.predict(vec)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"review": data, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
