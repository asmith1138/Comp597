import pandas as pd
import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import seaborn as sns
#import sklearn
#import csv
import os

#global trainData
trainData = np.array([[0,"This movie was bad"]])
testData = np.array([[0,"This movie was bad"]])
#myfile = 'filename.txt'
def buildData(trainData, testData):    
    for root, dirs, files in os.walk("aclImdb"):
        print(root)
        if root == "aclImdb/train/neg":
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/train/pos":
            for file in files:
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[1,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/test/neg":
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        testData=np.append(testData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclImdb/test/pos":
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
        if root == "aclImdb\\train\\neg":
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
        elif root == "aclImdb\\train\\pos":
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
        elif root == "aclImdb\\test\\neg":
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
        elif root == "aclImdb\\test\\pos":
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
#print(trainData)
#data_pd = pd.DataFrame(trainData)
#print (data_pd)

for root, dirs, files in os.walk("aclImdb"):
        print(root)
        if root == "aclImdb\\test\\neg":
            for file in files:
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        testData=np.append(testData,[[0,data]],axis=0)
                        print(file)
                        text_file.close()
                    except:
                        pass
            #print(trainData)

#data_pd = pd.DataFrame(trainData)
test_data_pd = pd.DataFrame(testData)
#data_pd.to_csv("aclimdb\\train.csv", header=None, index=None,mode='a')
test_data_pd.to_csv("aclimdb\\test.csv", header=None, index=None,mode='a')