import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

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
