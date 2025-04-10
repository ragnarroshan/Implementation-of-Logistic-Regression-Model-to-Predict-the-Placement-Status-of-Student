# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 
~~~
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop
~~~
## Program & Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: j kirthick roshan
RegisterNumber: 212223040097
*/
```
~~~
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head(5)
~~~
![image](https://github.com/user-attachments/assets/cd89d34d-6c2a-420e-a0db-8be612c81842)

~~~
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
~~~
![image](https://github.com/user-attachments/assets/08bab0f2-9ed6-4ba9-bdf5-2ce81b451040)

~~~
data1.isnull().sum()
~~~
![image](https://github.com/user-attachments/assets/c4ab9a49-2dfb-47bc-bfe1-2621f7989735)

~~~
data1.duplicated().sum()
~~~
![image](https://github.com/user-attachments/assets/e45217f7-9d5a-46c1-acd9-a32af7cad022)

~~~
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
~~~
![image](https://github.com/user-attachments/assets/410538ce-2548-4121-945b-c7676e121438)
~~~
x=data1.iloc[:,:-1]
x
~~~
![image](https://github.com/user-attachments/assets/15e18992-c462-4c49-9de4-81cb14656dd9)
~~~
y=data1["status"]
y
~~~
![image](https://github.com/user-attachments/assets/f1985f00-5659-4539-8b0a-953ae6d209f7)
~~~
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
~~~
~~~
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
~~~
![image](https://github.com/user-attachments/assets/c19409d8-0e9c-45d8-8193-78be87d0ed40)
~~~
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)
~~~
![image](https://github.com/user-attachments/assets/31e3ef80-2575-40f7-a9be-99d5efc8fc48)
~~~
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix
~~~
![image](https://github.com/user-attachments/assets/f8a757fc-b73f-4082-b9bb-a3f7426a9cff)
~~~
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
~~~
![image](https://github.com/user-attachments/assets/72216b41-de4a-4c4d-acf8-c19c99c7db42)
~~~
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~
![image](https://github.com/user-attachments/assets/4b3cf775-6cca-4943-8623-0fd09d8137cc)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
