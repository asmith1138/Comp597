#import pandas as pd
import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import seaborn as sns
#import sklearn
#import csv
import os
#from keras.models import Sequential
#from keras.layers import Dense

#global trainData
trainData = np.array([[0,"This movie was bad"]])
testData = np.array([[0,"This movie was bad"]])
#myfile = 'filename.txt'
def buildData(trainData, testData):    
    for root, dirs, files in os.walk("aclimdb"):
        print(root)
        if root == "aclimdb\\train\\neg":
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclimdb\\train\\pos":
            for file in files:
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        trainData=np.append(trainData,[[1,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclimdb\\test\\neg":
            for file in files:
                #print(file)
                with open(os.path.join(root, file), "r") as text_file:
                    try:
                        data = text_file.read()
                        testData=np.append(testData,[[0,data]],axis=0)
                        text_file.close()
                    except:
                        pass
        elif root == "aclimdb\\test\\pos":
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
    for root, dirs, files in os.walk("aclimdb"):
        if root == "aclimdb\\train\\neg":
            i = 0
            for file in files:
                if i < 500:    
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[0,data]],axis=0)
                            print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            print(trainData)
        elif root == "aclimdb\\train\\pos":
            i = 0
            for file in files:
                if i < 500:
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[1,data]],axis=0)
                            print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            print(trainData)
        elif root == "aclimdb\\test\\neg":
            i = 0
            for file in files:
                if i < 100:    
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            trainData=np.append(trainData,[[0,data]],axis=0)
                            print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            print(testData)
        elif root == "aclimdb\\test\\pos":
            i = 0
            for file in files:
                if i < 100:
                    with open(os.path.join(root, file), "r") as text_file:
                        try:
                            data = text_file.read()
                            testData=np.append(testData,[[1,data]],axis=0)
                            print(file)
                            text_file.close()
                        except:
                            pass
                i += 1
            print(testData)
    return trainData, testData



trainData, testData = buildDataSmall(trainData, testData)
#trainData, testData = buildData(trainData, testData)

#data_pd = pd.DataFrame(trainData)
#print (data_pd)

'''
#data = np.random.random((1000,100))
#labels = np.random.randint(2,size=(1000,1))
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=trainData.shape[1]))
model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(trainData,["Review","GoodOrBad"],epochs=10,batch_size=32)


#model.output_shape
#model.summary()
#model.get_config()
#model.get_weights()

predictions = model.predict(testData)
print(predictions)
'''