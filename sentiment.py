# Sentiment Analysis using NLP

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (you can replace with CSV later)
data = pd.DataFrame({
    'review': ['good product', 'bad service', 'excellent', 'worst experience', 'nice quality'],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
})

# Features and target
X = data['review']
y = data['sentiment']

# TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
