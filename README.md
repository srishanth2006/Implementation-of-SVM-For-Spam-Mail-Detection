# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SRISHANTH J
RegisterNumber:  212223240160
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
data.head()
<img width="551" alt="image" src="https://github.com/srishanth2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150319470/18fba85c-0485-43a1-8e98-8122a1e949ac">
data.info()
<img width="359" alt="image" src="https://github.com/srishanth2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150319470/52d77549-4870-4aec-9a3f-eb06a244ef0a">
data.isnull().sum()
<img width="170" alt="image" src="https://github.com/srishanth2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150319470/8e350baa-6098-4879-87ea-a519f869424d">
y_pred
<img width="573" alt="image" src="https://github.com/srishanth2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150319470/abbc6af9-e895-466e-af26-0d57078ace64">
accuracy
<img width="577" alt="image" src="https://github.com/srishanth2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150319470/5bc105dc-e500-4a88-896d-2f67a3a824fc">



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
