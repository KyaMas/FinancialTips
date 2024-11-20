import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load('./data/random_forest_model.pkl')
vectorizer = joblib.load('./data/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'review' not in data:
            return jsonify({"error": "Invalid input, 'review' key is required"}), 400

        review = data.get('review', '')

        # Preprocess the review using the loaded vectorizer
        review_vector = vectorizer.transform([review])

        # Ensure the transformed input matches the expected feature count
        if review_vector.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature mismatch: expected {model.n_features_in_} features, but got {review_vector.shape[1]}"}), 400

        # Make prediction
        prediction = model.predict(review_vector)

        return jsonify({
            'review': review,
            'recommendation': int(prediction[0])
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong on the server"}), 500

if __name__ == '__main__':
    app.run(debug=True)
