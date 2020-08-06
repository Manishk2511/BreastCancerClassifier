from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('breast_cancer.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test,y_pred)
accuracies = cross_val_score(classifier,x_train,y_train,cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))