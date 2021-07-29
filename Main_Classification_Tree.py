import numpy as np
import matplotlib.pyplot as plt
import csv

from numpy import genfromtxt
from sklearn.model_selection import train_test_split

from class_tree import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


my_data = genfromtxt('df_data.csv', delimiter=',')
X = np.ones((np.size(my_data,axis=0)-1,3))
y = 3*np.ones((np.size(my_data,axis=0)-1,1))
for i in range(0,(np.size(my_data,axis=0)-1)):
        X[i][:]=my_data[i+1][0:3]
    
with open('df_data.csv') as csvfile: 
        my_data_2 = csv.reader(csvfile)  
        i=0
        next(my_data_2)
        for line in my_data_2:
            if line[3]== 'med':
                y[i]=0
            elif line[3]=='low':
                y[i]=2
            else:
                y[i]=1
            i=i+1

y = y[:,0]
y = np.int64(y)
# y = map(int,y)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2)  
clf = DecisionTree(max_depth=10)
clf.fit(X1_train,y1_train)

y1_pred = clf.predict(X1_train)
y1_pred = np.int64(y1_pred)
acc = accuracy(y1_train,y1_pred)
print("Accuracy:", acc)