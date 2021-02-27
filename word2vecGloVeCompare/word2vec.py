import pandas as pd
import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import math
#import seaborn as sns
#import sklearn
#import csv
import os
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from gensim.models.word2vec import Word2Vec



#Using Word2Vec what?
with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

#training_sentences = DataPrep.train_news['Statement']

#model = Word2Vec(training_sentences, size=100) # x be tokenized text
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))

#idk what this is for...
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, size=100):
        self.word2vec = word2vec
        self.dim = size

    def fit(self, X, y): # what are X and y?
        return self

    def transform(self, X): # should it be training_sentences?
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



trainData = pd.read_csv('train.csv').values
testData = pd.read_csv('test.csv').values

model = [('log_reg',LogisticRegression()),('logcv_reg',LogisticRegressionCV())]

nltk.download("stopwords","data")
nltk.download("punkt")
nltk.data.path.append("data")

charfilter = re.compile('[a-zA-Z]+')

stopWords = stopwords.words('english')

def SimpleTokenizer(text):
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    ntokens = list(filter(lambda token: charfilter.match(token),tokens))
    return ntokens

print("building pipeline...")
pipeline = Pipeline([('vec', TfidfVectorizer(tokenizer=SimpleTokenizer)),('log_reg',LogisticRegression())])
print("fitting...")
y = trainData[:,0]
x = trainData[:,1]
y = y.astype('int')
pipeline.fit(x,y)

y_test = testData[:,0]
x_test = testData[:,1]
y_test = y_test.astype('int')
print("predicting...")
test_y = pipeline.predict(x_test)
#print(pipeline.decision_function)
print(classification_report(y_test,test_y, digits=3))
