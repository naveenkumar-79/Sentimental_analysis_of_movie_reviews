# Sentimental_analysis_of_movie_reviews

📌**Project Overview**

This project is a Sentiment Analysis system for Movie Reviews built using Natural Language Processing (NLP) and Deep Learning techniques. The model analyzes textual movie reviews and predicts whether the sentiment expressed is Positive or Negative.

The system uses a Bidirectional Simple RNN deep learning model and demonstrates a complete NLP pipeline including text preprocessing, tokenization, vectorization, and sentiment prediction.

🚀 **Features**

Classifies movie reviews as Positive or Negative

Uses real-world IMDB movie review dataset

Text preprocessing: cleaning, stopword removal, lemmatization

Deep learning–based sentiment prediction

Model loading using Pickle

Modular, class-based Python implementation

Logging and exception handling

🛠 **Technologies & Libraries Used**

Python 3

Pandas & NumPy – Data handling

NLTK – Text preprocessing and lemmatization

TensorFlow / Keras – Deep learning model

Scikit-learn – Supporting utilities

Matplotlib – Visualization (optional)

Pickle – Model serialization

🧠 **How It Works**

Loads the IMDB movie reviews dataset

Cleans the input review text (lowercasing, punctuation removal)

Removes stopwords and applies lemmatization

Converts text into numerical form using one-hot encoding

Pads sequences to a fixed length

Loads the pre-trained sentiment analysis model

Predicts sentiment as Positive or Negative

📂 **Project Structure**

sentiment-analysis-movie-reviews/
│
├── main.py                 # Main application file
├── analysis.pkl            # Trained sentiment analysis model
├── IMDB Dataset.csv        # Dataset file
├── log.py                  # Logging configuration
├── README.md               # Project documentation
▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install numpy pandas nltk tensorflow scikit-learn matplotlib
2️⃣ Download NLTK Resources
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
3️⃣ Run the Application
python main.py

📌 **Sample Input**

review = "This movie was absolutely amazing with great performances"
📄 Sample Output
Prediction of the review: positive
🎯 **Use Cases**

Movie review analysis

Opinion mining

Recommendation systems

NLP learning and experimentation

📈 **Learning Outcomes**

Understanding NLP preprocessing pipelines

Hands-on experience with sentiment analysis

Working with deep learning text models

Model loading and inference using TensorFlow

Applying lemmatization and stopword removal

🔮 **Future Enhancements**

Build and train the model within the project

Add web interface using Flask or Streamlit

Support multi-class sentiment (rating-based)

Improve accuracy using LSTM / Bi-LSTM / Transformers

🤝 **Contributing**

Contributions, suggestions, and improvements are welcome!

📬 Contact

Name: P.Naveen Kumar

🔗 LinkedIn: www.linkedin.com/in/naveenkumar-puppala-b87737332

🐙 Gmail: puppalanaveenkumar11@gmail.com
