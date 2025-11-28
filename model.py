import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels to binary
data = data.dropna(subset=['Message'])
data['Message'] = data['Message'].fillna('')
data = data[['Message', 'Spam/Ham']]
data = data.rename(columns={'Spam/Ham':'label'})

data['label'] = data['label'].map({'ham':0, 'spam':1})


X = data['Message']
y = data['label']

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", ngram_range=(1,3))),
    ('clf', LogisticRegression(max_iter=500))
])


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())


print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("\nModel trained and saved as model.pkl")
