import pandas as pd
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import math
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
# import nltk
# import re
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk import PorterStemmer
from zeugma.embeddings import EmbeddingTransformer






trainData = pd.read_csv('train.csv').values
testData = pd.read_csv('test.csv').values


#incase I wanted to do all 3
#model = [('log_reg',LogisticRegression()),('logcv_reg',LogisticRegressionCV()),('MLP_Class',MLPClassifier())]
model = LogisticRegression()
'''
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
'''

print("Vectorizing?...")
glove = EmbeddingTransformer('glove')
#print("building pipeline...")
#pipeline = Pipeline([('vec', TfidfVectorizer(tokenizer=GloVeTransformer)),('log_reg',LogisticRegression())])

print("fitting...")
y = trainData[:,0]
x = glove.transform(trainData[:,1])
y = y.astype('int')
#pipeline.fit(x,y)
model.fit(x,y)

y_test = testData[:,0]
x_test = testData[:,1]
y_test = y_test.astype('int')
print("predicting...")
#test_y = pipeline.predict(x_test)
test_y = model.predict(x_test)
print(classification_report(y_test,test_y, digits=3))



#glove = EmbeddingTransformer('glove')
#x_train = glove.transform(corpus_train)

#model = LogisticRegression()
#model.fit(x_train, y_train)

#x_test = glove.transform(corpus_test)
#model.predict(x_test)
