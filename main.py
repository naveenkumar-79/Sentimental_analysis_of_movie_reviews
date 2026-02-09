import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
nltk.download('punkt',quiet=True)
nltk.download('wordnet')
nltk.download('stopwords')
import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from log import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
class sent_analysis:
    def __init__(self):
        self.df=pd.read_csv(r'D:\NLP_projects\sentimental_analysis\IMDB Dataset.csv')
        logger.info(self.df.head(5))
        # logger.info(self.df['sentiment'].value_counts())
    def model_loading(self):
        try:
            with open('analysis.pkl', 'rb') as f:
                self.m = pickle.load(f)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Performance Error : {er_lin.tb_lineno} : due to {er_msg}")
    def model_testing(self):
        labels = ['positive', 'negative']
        dic_size = 10000
        review = ['I was very lucky to see this film as part of the Melbourne International Film Festival 2005 only a few days ago. I must admit that I am very partial to movies that focus on human relations and especially the ones which concentrate on the tragic side of life. I also love the majority of Scandinavian cinematic offerings, there is often a particular deep quality in the way the story unfolds and the characters are drawn. Character building in this film is extraordinary in its details and its depth. This is despite the fact that we do encounter quite a number of characters all with very particular personal situations and locations within their community. The audience at the end of the screening was very silent and pensive. I am still playing some of those scenes in my mind and I am still amazed at their power and meaningfulness.']
        text = review[0].lower()
        text = ''.join([i for i in text if i not in string.punctuation])
        text = ' '.join([lemma.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])
        v = [one_hot(i, dic_size) for i in [text]]
        p = pad_sequences(v, maxlen=953, padding='post')
        logger.info(f'prediction of the review: {(labels[np.argmax(self.m.predict(p))])}')

if __name__ == "__main__":
    obj=sent_analysis()
    obj.model_loading()
    obj.model_testing()