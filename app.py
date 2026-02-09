from flask import Flask, render_template, request
import pickle
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================
with open("analysis.pkl", "rb") as f:
    model = pickle.load(f)

# MUST MATCH TRAINING
dic_size = 10000
maxlen = 953

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    return text


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review = None

    if request.method == "POST":
        review = request.form.get("review")

        cleaned = clean_text(review)

        encoded = [one_hot(cleaned, dic_size)]
        padded = pad_sequences(encoded, maxlen=maxlen, padding="post")

        # 🔥 SAFE PREDICTION
        pred = model.predict(padded)

        # Convert to float safely
        score = float(np.squeeze(pred))

        print("Prediction score:", score)  # DEBUG

        if score >= 0.5:
            prediction = "Positive Review 😊"
        else:
            prediction = "Negative Review 😞"

    return render_template(
        "index.html",
        prediction=prediction,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)
