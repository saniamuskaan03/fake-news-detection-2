from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv("News.csv")

#data preprocessing
df.drop_duplicates(subset=["title", "text"], inplace=True)
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["content"] = df["title"] + " " + df["text"]
df = df.dropna(subset=["content", "class"])
#labels
X = df["content"]
y = df["class"]  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2),
    max_features=10000
)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)
model.fit(X_train, y_train)

# Evaluation using metrics
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

@app.route('/')
def home():
    return open('index.html').read()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("news", "")
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({"prediction": "REAL" if pred == 1 else "FAKE"})

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
