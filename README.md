# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 'pandas' is used for handling the dataset
2. numpy is used for numerical operations and matrix computations.
3. The dataset is loaded from a CSV file.
4. Unnecessary columns ('sl_no' and 'salary') are removed as they do not contribute to placement prediction.
5. The categorical columns are converted into category data type.
6. Each categorical variable is assigned numerical codes for machine learning compatibility.
7.'X' → Independent variables (features).
8. 'Y' → Dependent variable (placement status: 1 = Placed, 0 = Not Placed).
9. 'theta' (weights) is initialized randomly with the same number of elements as the number of features in 'X'.
10. The sigmoid function is used to output probabilities between 0 and 1, which helps in classification.
11. The log loss function is used for evaluating the performance of logistic regression.
12. The model iteratively updates 'theta' using gradient descent to minimize the loss function.
13. Learning rate ('alpha') controls step size.
14. The loop runs for 'num_iterations' to optimize 'theta'.
15. The model is trained using 1000 iterations with a learning rate of 0.01.
16. If ℎ(𝑥)≥0.5, classify as 1 (Placed); otherwise, 0 (Not Placed).
17. Accuracy is calculated by comparing predictions ('y_pred') with actual values ('y').
18. 'y_pred' → Predicted placement results.
19. 'y' → Actual placement results.
20. The model predicts whether a student with given features will be placed.
21. Another new student is tested to check placement prediction
## Program:

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: D.Varshini
RegisterNumber: 212223230234

 import pandas as pd
import numpy as np

dataset = pd.read_csv('Placement_Data.csv')
print("Name: Varshini D\nReg.no: 212223230234)
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, Y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient  = X.T.dot(h - y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict (theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred  = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)

print(y_pred)
print(y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0 ]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```


## Output:

![image](https://github.com/user-attachments/assets/7f3f3f0c-82ec-4f5b-b2e1-6844d4c87c6c)
![image](https://github.com/user-attachments/assets/85c1182f-c710-4d1a-89fa-4e656e54c490)
![image](https://github.com/user-attachments/assets/21a38af0-d4ca-4e6f-aeff-84e1228cdea6)
![image](https://github.com/user-attachments/assets/2fbac6e4-833f-48cf-b34d-02cb03334c27)
![image](https://github.com/user-attachments/assets/e038bc43-e5d5-446a-84c5-1e12bf576351)
![image](https://github.com/user-attachments/assets/d279efe5-233e-4bfa-8b9a-81351a7bede6)
![image](https://github.com/user-attachments/assets/64d6a26b-2f60-402e-addd-df60b8e337d5)
![image](https://github.com/user-attachments/assets/da127623-0650-4034-8362-716e1115ffbd)
![image](https://github.com/user-attachments/assets/fc3473e1-ea05-42cf-90b4-ae4209f40b71)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
