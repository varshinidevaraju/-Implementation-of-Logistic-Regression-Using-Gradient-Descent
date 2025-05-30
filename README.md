# Ex.No-06-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.
 ```
Developed by: Varshini D
RegisterNumber:  212223230234
```

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 

6.Define a function to predict the Regression value.
```
## Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset
```
## Output:
![image](https://github.com/user-attachments/assets/13598c70-0ee8-46e2-9487-36b8a3ec9747)
```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

![image](https://github.com/user-attachments/assets/31f80549-1cd3-46b1-b660-2d800ec42eae)

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset
```
![image](https://github.com/user-attachments/assets/9bea160c-8de2-4cf2-a0f2-12613af451f4)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
![image](https://github.com/user-attachments/assets/769482d6-298e-433d-b0ab-62f67f1b46a6)
```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/8b571dc1-5ede-47da-9df9-7ac8782cfdb0)
```
print(y_pred)
```
![image](https://github.com/user-attachments/assets/4a33f83b-dd60-4b8f-b984-1f2968540e77)
```
print(Y)
```
![image](https://github.com/user-attachments/assets/63d79797-923c-4242-bf9d-e5ea2b013e2d)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/481a16ab-0007-42cd-b2aa-fbdb9bdb681d)
```
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/24e13c63-7694-43e4-b599-2153d9733e36)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
