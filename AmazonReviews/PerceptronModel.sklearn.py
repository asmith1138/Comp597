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
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

#global trainData
trainData = np.array([[0,"This movie was bad"]])
testData = np.array([[0,"This movie was bad"]])
#these took too long, so i loaded them into csvs
def buildData(trainData, testData):    
    for root, dirs, files in os.walk("aclImdb"):
        print(root)
        if root == "aclImdb/train/neg" or root == "aclImdb\\train\\neg" :
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/train/pos" or root == "aclImdb\\train\\pos":
            for file in files:
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[1,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/test/neg" or root == "aclImdb\\test\\neg" :
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        testData=np.append(testData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/test/pos" or root == "aclImdb\\test\\pos":
            for file in files:
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        testData=np.append(testData,[[1,data]],axis=0)
                        text_file.close()
                    except:
                        pass
    return trainData, testData

def buildDataSmall(trainData, testData):    
    for root, dirs, files in os.walk("aclImdb"):
        print(root)
        if root == "aclImdb/train/neg" or root == "aclImdb\\train\\neg" :
            i = 0
            for file in files:
                if i < 500:    
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[0,data]],axis=0)
                            #print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            #print(trainData)
        elif root == "aclImdb/train/pos" or root == "aclImdb\\train\\pos":
            i = 0
            for file in files:
                if i < 500:
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[1,data]],axis=0)
                            #print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            #print(trainData)
        elif root == "aclImdb/test/neg" or root == "aclImdb\\test\\neg" :
            i = 0
            for file in files:
                if i < 100:    
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[0,data]],axis=0)
                            #print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            #print(testData)
        elif root == "aclImdb/test/pos" or root == "aclImdb\\test\\pos":
            i = 0
            for file in files:
                if i < 100:
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            testData=np.append(testData,[[1,data]],axis=0)
                            #print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            #print(testData)
    return trainData, testData



#trainData, testData = buildDataSmall(trainData, testData)
#trainData, testData = buildData(trainData, testData)
trainData = pd.read_csv('aclimdb\\train.csv').values
#trainData = np.genfromtxt('aclimdb\\train.csv',delimiter=',')
testData = pd.read_csv('aclimdb\\test.csv').values
#testData = np.genfromtxt('aclimdb\\test.csv',delimiter=',')
#print(trainData)
#data_pd = pd.DataFrame(trainData)
#print (data_pd)

#incase I wanted to do all 3
model = [('log_reg',LogisticRegression()),('logcv_reg',LogisticRegressionCV()),('MLP_Class',MLPClassifier())]

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
pipeline = Pipeline([('vec', TfidfVectorizer(tokenizer=SimpleTokenizer)),('MLP_Class',MLPClassifier())])

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
