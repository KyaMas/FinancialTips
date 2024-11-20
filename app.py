from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, origins="*")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/test1', methods=['GET'])
def test1():

    filename = secure_filename(request.args.get('file'))
    try:
        if filename and allowed_filename(filename):
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as f:
                return f.read()
    except IOError:
        pass
    return "Unable to read file"

    # Load dataset
    df = pd.read_csv('./data/ecommerce_reviews.csv')

    # Handle missing values
    df['Review Text'] = df['Review Text'].fillna('')  # Fill missing reviews with empty strings
    df['Title'] = df['Title'].fillna('')  # Fill missing titles with empty strings

    # Clean text
    def clean_text(text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text

    df['Review Text'] = df['Review Text'].apply(clean_text)
    df['Title'] = df['Title'].apply(clean_text)

    # Save cleaned data
    df.to_csv('./data/cleaned_reviews.csv', index=False)
    print("Cleaned data saved to ./data/cleaned_reviews.csv")

    # Visualize rating distribution
    sns.countplot(x='Rating', data=df)
    plt.title('Rating Distribution')
    plt.savefig('./data/rating_distribution.png')

    # Generate a word cloud
    text = " ".join(review for review in df['Review Text'])
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('./data/review_wordcloud.png')
    return jsonify({'message': 'Hello, World!'})

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