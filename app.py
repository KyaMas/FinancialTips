from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score
import joblib


from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, origins="*")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/test1', methods=['POST'])
def test1():

    if 'file' not in request.files:
        return jsonify({'message': 'Invalid File'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'Select the File'})
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        train_model(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return jsonify({'message': 'Model trained successfully!'})

    # # Handle missing values
    # df['Review Text'] = df['Review Text'].fillna('')  # Fill missing reviews with empty strings
    # df['Title'] = df['Title'].fillna('')  # Fill missing titles with empty strings

    # # Clean text
    # def clean_text(text):
    #     text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    #     text = text.lower()  # Convert to lowercase
    #     return text

    # df['Review Text'] = df['Review Text'].apply(clean_text)
    # df['Title'] = df['Title'].apply(clean_text)

    # # Save cleaned data
    # df.to_csv('./data/cleaned_reviews.csv', index=False)
    # print("Cleaned data saved to ./data/cleaned_reviews.csv")

    # # Visualize rating distribution
    # sns.countplot(x='Rating', data=df)
    # plt.title('Rating Distribution')
    # plt.savefig('./data/rating_distribution.png')

    # # Generate a word cloud
    # text = " ".join(review for review in df['Review Text'])
    # wordcloud = WordCloud(background_color="white").generate(text)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.savefig('./data/review_wordcloud.png')
    # return jsonify({'message': 'Hello, World!'})

def train_model(file):
    # Load dataset
    df = pd.read_csv(file)

    # Ensure no missing values in 'Review Text'
    df['Review Text'] = df['Review Text'].fillna('')  # Replace NaN with empty strings

    # Feature engineering
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    text_features = tfidf.fit_transform(df['Review Text'])  # No more NaN issues

    numerical_features = df[['Age', 'Positive Feedback Count']]
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    X = hstack([text_features, scaled_numerical_features])
    y = df['Recommended IND']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(clf, './data/random_forest_model.pkl')
    joblib.dump(tfidf, './data/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")

    # Save the vectorizer
    joblib.dump(tfidf, './data/tfidf_vectorizer.pkl')
    print("Vectorizer saved!")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('error'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('error'))
    if file and file.filename.endswith('.csv'):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('success'))
    else:
        return redirect(url_for('error'))

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/submit_tip', methods=['POST'])
def submit_tip():
    user_input = request.form['user_input']
    # Handle the user input as needed
    return redirect(url_for('tips'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)