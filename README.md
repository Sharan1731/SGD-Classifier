# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm.
Step 1:
Import Necessary Libraries and Load Data.

Step 2:
Split Dataset into Training and Testing Sets.

Step 3:
Train the Model Using Stochastic Gradient Descent (SGD).

Step 4:
Make Predictions and Evaluate Accuracy.

Step 5:
Generate Confusion Matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SHARAN G
RegisterNumber:  212223230203
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![Screenshot 2024-09-19 161055](https://github.com/user-attachments/assets/bc4161d6-a438-482f-a8d5-704eda8a9a40)

![Screenshot 2024-09-19 161101](https://github.com/user-attachments/assets/cd437b3a-72bb-45ca-8e06-b0543370b8c8)

![Screenshot 2024-09-19 161105](https://github.com/user-attachments/assets/2568d2ca-d7ec-4fe4-97c7-e275a12b9546)

![Screenshot 2024-09-19 161110](https://github.com/user-attachments/assets/b3077b7d-7e71-4f2c-b61e-95bab99a4962)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
