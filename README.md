# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the tumor dataset containing medical test features such as radius, texture, perimeter, area, smoothness, and other diagnostic measurements.

2.Separate the dataset into input features (tumor characteristics) and the target class (benign or malignant), then preprocess the data by handling missing values and encoding class labels if required.

3.Split the dataset into training and testing sets using stratified sampling to maintain the class distribution.

4.Train a Decision Tree classifier on the training data to learn decision rules based on feature values.

5.Use the trained Decision Tree model to predict the tumor class for the testing dataset.

6.Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix to assess classification performance.

7.Use the trained model to classify new tumor samples as benign or malignant based on their lab test results.
## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: Shrivarshan")
print("Register Number: 25019111")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:
![alt text](<Screenshot 2026-03-10 141216.png>)
![alt text](<Screenshot 2026-03-10 141206.png>)


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
