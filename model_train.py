from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Load cleaned data
df = pd.read_csv('./data/cleaned_reviews.csv')

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


# problem statement 
# proposed solution 
# related work 
# methodology -> data set 
# data set cleaning & pre processing (19 products and 9 users)
# cleaned data -> we have user details with their spending of different item with the review and ratings (huggegin space -> distiler bert uncased finetuned on english)
# find the all products with sentiment rating from the review (NLP part)
# ML recommendation system -> based on user id, sentiment rating, avg sale price, cloth type
# input diff user id -> show them which product they can buy