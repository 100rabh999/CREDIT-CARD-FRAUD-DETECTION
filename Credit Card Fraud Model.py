#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the credit card fraud dataset 
data = pd.read_csv('C:\\Users\\madan\\OneDrive\\Desktop\\fraudTest.csv')

# Explore the dataset
print(data.head())

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['merchant', 'category', 'gender', 'job'])

# Features and target variable
X = data_encoded.drop('Is_fraud', axis=1)
y = data_encoded['Is_fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(model_name, y_true, y_pred):
    print(f"\n{model_name} Model:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, lr_predictions)
evaluate_model("Decision Tree", y_test, dt_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)


# In[ ]:


pip install numpy==1.24.5  # Replace with a version between 1.18.5 and 1.25.0


# In[ ]:


pip install --upgrade scipy


# In[ ]:




