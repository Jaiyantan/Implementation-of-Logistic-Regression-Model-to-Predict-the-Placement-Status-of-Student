# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset, drop unnecessary columns, and encode categorical variables. 
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets. 
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Jaiyantan S
RegisterNumber:  212224100021
*/
```

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:

HEAD

![image](https://github.com/user-attachments/assets/5e7f65b2-e4bf-4cc9-afd0-ad69877cf8f8)

COPY

![image](https://github.com/user-attachments/assets/7fcc3e09-2fdf-473e-8395-80058c78bf6a)

FIT TRANSFORM

![image](https://github.com/user-attachments/assets/5fc10b4f-30af-427e-9918-61da8017f402)

LOGISTIC REGRESSION

![image](https://github.com/user-attachments/assets/eb254409-0bb1-4b0e-ba64-e8a0197f0697)

ACCURACY SCORE

![image](https://github.com/user-attachments/assets/1a85a5eb-d8fb-48da-af8a-2e5679677d18)

CONFUSION MATRIX

![image](https://github.com/user-attachments/assets/1ed9485e-5063-4c42-8150-68be5513875e)

CLASSIFICATION REPORT & PREDICTION

![image](https://github.com/user-attachments/assets/b1b6ac3e-bd00-49c4-862f-6e7d5206aed7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
