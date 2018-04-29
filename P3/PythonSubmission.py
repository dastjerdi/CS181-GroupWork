import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa
import librosa.feature as lf
import librosa.display
import IPython.display as ipd
import numpy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

train_data_reader = pd.read_csv("train.csv", header=None, chunksize=200)
train_data_1 = train_data_reader.get_chunk(1)
train_data_1.head()
print(np.array(train_data_1)[0])

train_data = np.array([])

counter = 0
train_data = np.array([])
for chunk in train_data_reader:
    #print(chunk)
    chunk1 = np.array(chunk)
    for thing in chunk1:
        print(counter)
        thing1 = np.array(thing)
        #print(thing1)
        row = np.array([])
        cstft = np.mean(lf.chroma_stft(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, cstft))
        cqt = np.mean(lf.chroma_cqt(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, cqt))
        sens = np.mean(lf.chroma_cens(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, sens))
        spcent = np.mean(lf.spectral_centroid(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, spcent))
        flatness = np.mean(lf.spectral_flatness(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, flatness))
        rolloff = np.mean(lf.spectral_rolloff(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, rolloff))
        mspec = np.mean(lf.melspectrogram(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, mspec))
        mfcc = np.mean(lf.mfcc(thing1[:-1],n_mfcc=30).T, axis=0)
        row = np.concatenate((row,mfcc))
        tonnetz = np.mean(lf.tonnetz(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, tonnetz))
        rmse =  np.mean(lf.rmse(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, rmse))
        contrast =  np.mean(lf.spectral_contrast(thing1[:-1]).T, axis=0)
        row = np.concatenate((row, contrast))
        tempo =  np.mean(lf.tempogram(thing[:-1], win_length=88).T, axis=0)
        row = np.concatenate((row, tempo))
        row = np.append(row, thing1[-1])
        #print(len(row))

        train_data = np.append(train_data, row)
        counter += 1

columns = ["feat_" + str(i) for i in range(299)]
columns.append("class")
df_train2 = pd.DataFrame(columns=columns)

for i in range(6325):
    print(float(i)/6325. * 100)
    row = train_data[300*i:300*(i+1)]
    #print(pd.Series(row))
    df_train2.loc[i] = row
    #print(df_train)

df_train2.to_csv("df_train2.csv")

train_data_cstft  = pd.DataFrame(lf.chroma_stft(np.array(train_data_1)[0]))
train_data_cstft.shape

train_data_spcent = pd.DataFrame(lf.spectral_centroid(np.array(train_data_1)[0]))
train_data_spcent.head()

test_data_reader = pd.read_csv("test.csv", header=None, chunksize=200)

counter = 0
counter = 0
test_data = np.array([])
for chunk in test_data_reader:
    #print(chunk)
    chunk1 = np.array(chunk)
    for thing in chunk1:
        print(counter)
        thing1 = np.array(thing)
        #print(thing1)
        row = np.array([thing1[0]])
        cstft = np.mean(lf.chroma_stft(thing1[1:]).T, axis=0)
        row = np.concatenate((row, cstft))
        cqt = np.mean(lf.chroma_cqt(thing1[1:]).T, axis=0)
        row = np.concatenate((row, cqt))
        sens = np.mean(lf.chroma_cens(thing1[1:]).T, axis=0)
        row = np.concatenate((row, sens))
        spcent = np.mean(lf.spectral_centroid(thing1[1:]).T, axis=0)
        row = np.concatenate((row, spcent))
        flatness = np.mean(lf.spectral_flatness(thing1[1:]).T, axis=0)
        row = np.concatenate((row, flatness))
        rolloff = np.mean(lf.spectral_rolloff(thing1[1:]).T, axis=0)
        row = np.concatenate((row, rolloff))
        mspec = np.mean(lf.melspectrogram(thing1[1:]).T, axis=0)
        row = np.concatenate((row, mspec))
        mfcc = np.mean(lf.mfcc(thing1[1:],n_mfcc=30).T, axis=0)
        row = np.concatenate((row,mfcc))
        tonnetz = np.mean(lf.tonnetz(thing1[1:]).T, axis=0)
        row = np.concatenate((row, tonnetz))
        rmse =  np.mean(lf.rmse(thing1[1:]).T, axis=0)
        row = np.concatenate((row, rmse))
        contrast =  np.mean(lf.spectral_contrast(thing1[1:]).T, axis=0)
        row = np.concatenate((row, contrast))
        tempo =  np.mean(lf.tempogram(thing1[1:], win_length=88).T, axis=0)
        row = np.concatenate((row, tempo))
        #row = np.append(row, thing1[-1])
        #print(len(row))

        test_data = np.append(test_data, row)
        counter += 1

columns = ["Id"] + ["feat_" + str(i) for i in range(299)]
df_test2 = pd.DataFrame(columns=columns)

for i in range(1000):
    print(float(i)/1000. * 100)
    row = test_data[300*i:300*(i+1)]
    #print(pd.Series(row))
    df_test2.loc[i] = row
    #print(df_train)

df_test2.to_csv("df_test2.csv")

df_train2 = pd.read_csv("df_train2.csv")
df_test2 = pd.read_csv("df_test2.csv")

y_train = df_train2["class"]
X_train = df_train2.drop("class", 1).drop("Unnamed: 0",1)
X_test = df_test2.drop("Id",1).drop("Unnamed: 0",1)
ids = df_test2["Id"]

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rfpreds = pd.DataFrame(columns=["Id", "Prediction"])
column1 = np.array(ids).astype(int)
column2 = pd.Series(y_pred)
print(len(column2))
rfpreds["Prediction"] = column2
rfpreds["Id"] = pd.to_numeric(column1, downcast="integer")
rfpreds.to_csv("rfpreds1.csv", index=False)

rf2 =RandomForestClassifier()
cvdict = {"max_depth":range(1,20,2), "n_estimators": range(1,10), "min_samples_split":range(2,5)}
cvrf = GridSearchCV(rf, cvdict)
cvrf.fit(X_train, y_train)
y_pred2 = cvrf.predict(X_test)

rf2preds = pd.DataFrame(columns=["Id", "Prediction"])
column1 = np.array(ids).astype(int)
column2 = pd.Series(y_pred2)
print(len(column2))
rf2preds["Prediction"] = column2
rf2preds["Id"] = pd.to_numeric(column1, downcast="integer")
rf2preds.to_csv("rfpreds2.csv", index=False)

svm = GridSearchCV(SVC(), {"C":[float(i) for i in range(1,10,2)], "kernel": ["linear", "poly", "rbf", "sigmoid"]})
svm.fit(X_train, y_train)
y_pred3 = svm.predict(X_test)

svpreds = pd.DataFrame(columns=["Id", "Prediction"])
column1 = np.array(ids).astype(int)
column2 = pd.Series(y_pred3)
print(len(column2))
svpreds["Prediction"] = column2
svpreds["Id"] = pd.to_numeric(column1, downcast="integer")
svpreds.head()
svpreds.to_csv("svpreds.csv", index=False)

ada = AdaBoostClassifier(base_estimator=RandomForestClassifier())
ada.fit(X_train, y_train)
y_pred5 = ada.predict(X_test)

adapreds = pd.DataFrame(columns=["Id", "Prediction"])
column1 = np.array(ids).astype(int)
column2 = pd.Series(y_pred5)
print(len(column2))
adapreds["Prediction"] = column2
adapreds["Id"] = pd.to_numeric(column1, downcast="integer")
adapreds.to_csv("adapreds2.csv", index=False)

train = pd.read_csv("df_train2.csv")
test = pd.read_csv("df_test2.csv")
train = train.drop(['Unnamed: 0'], axis = 1)
test = test.drop(['Unnamed: 0', 'Id'], axis = 1)

y_train = train['class']
x_train = train.drop(['class'], axis = 1)

LogReg = LogisticRegressionCV(multi_class='multinomial', solver='sag')
LogReg.fit(x_train, y_train)
predictions = LogReg.predict(test)

predictions = pd.Series(predictions)
LogRegCV = pd.DataFrame(predictions, columns=['Prediction'])
LogRegCV.Prediction = LogRegCV.Prediction.astype(int)
LogRegCV.to_csv("LogRegCV.csv", index_label=['Id'])

NN = MLPClassifier()
NN.fit(x_train, y_train)
predictions = NN.predict(test)
predictions = pd.Series(predictions)
NNPred = pd.DataFrame(predictions, columns=['Prediction'])
NNPred.Prediction = NNPred.Prediction.astype(int)
NNPred.to_csv("NNPred.csv", index_label=['Id'])

GNB = GaussianNB()
predictions = GNB.fit(x_train, y_train).predict(test)
predictions = pd.Series(predictions)
GNBPred = pd.DataFrame(predictions, columns=['Prediction'])
GNBPred.Prediction = GNBPred.Prediction.astype(int)
GNBPred.to_csv("GNBPred.csv", index_label=['Id'])
