import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import csv
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer




k=input("Enter the name of the file:")
df=pd.read_csv(k)
df.dropna(inplace=True)
tweet=[ i for i in df['Tweets']]
d1={'positive':1,'neutral':0,'negative':-1}
df['Sentiment']=df['Sentiment'].map(d1)
l=[] #store accuracy of all the models
name=k[:-4]
l.append(name)

''' Used for forming feature vectors through bag of words technique'''
cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',max_features = 1500)
X = cv.fit_transform(tweet).toarray()
y = df.iloc[:, 1].values
X_train,X_test,Y_train,Y_test= train_test_split(X,y,train_size=0.7,random_state=42)
     
def print_score(clf,X_train,Y_train,X_test, Y_test,train=True):
    if train:
        print("\nTraining Result:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_train,clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train,clf.predict(X_train))))
        
        res=cross_val_score(clf,X_train,Y_train,cv=10,scoring="accuracy")
        print("Average Accuracy:\t {0:.4f}".format(np.mean(res)))
        print ("Accuracy SD:\t\t {0:.4f}".format(np.std(res)))
    
    elif train==False:
        print("\nTest Results:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_test,clf.predict(X_test))))
        print("\nConfusion Matrix:\n{}\n".format(confusion_matrix(Y_test,clf.predict(X_test))))
        return "{0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test)))

""" CNN """
"""
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)
print("\n\n\t\t----- CNN Details-------\n\n")
model.predict(X_test[:4])

"""

""" Deep NN """
model = Sequential()
embedding_layer = Embedding(10, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

"""KNN"""
knn= KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train,Y_train)
print ("\n \t\t------KNN Classifier----\n")       
#Scores for training data
print_score(knn,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
t=print_score(knn,X_train,Y_train,X_test, Y_test,train=False)
l.append(t)

'''SVM'''
clf=svm.SVC(kernel='rbf', degree=3,  gamma=0.7)
clf.fit(X_train,Y_train)

print("\n\n\t\t----- SVM Details-------\n\n")
#Scores for training data
print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
t=print_score(clf,X_train,Y_train,X_test, Y_test,train=False)
l.append(t)

"""XG BOOST"""
import xgboost as xgb
clf=xgb.XGBClassifier()
clf.fit(X_train,Y_train)
print("\n\n XG BOOST\n\n")
#Scores for training data
print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
t=print_score(clf,X_train,Y_train,X_test, Y_test,train=False)
l.append(t)


""" Naive Bayes"""
model = GaussianNB()
model.fit(X_train,Y_train)
print("\n\n Naive Bayesian\n\n")
#Scores for training data
print_score(model,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
t=print_score(model,X_train,Y_train,X_test, Y_test,train=False)
l.append(t)

''' Append the accuracy to the scores file for further analysis''' 
with open("Scores.csv", 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow((l[0], l[1],l[2],l[3],l[4]))
    
# Read the datafile and plot 
""" For different colleges different ML techniques perform differently"""
dg=pd.read_csv("Scores.csv")
dg.plot(x='College', y=["KNN","SVM","XGBoost","NBayes"], kind="bar")
