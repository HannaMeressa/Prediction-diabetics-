# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:03:12 2024

@author: Hani
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
df = pd.read_csv('diabetes.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file

# Feature encoding
df['diabetes_type'] = df['diabetes_type'].map({'prediabetes': 0, 'type1': 1, 'type2': 2})
df['treatment'] = df['treatment'].map({'lifestyle': 0, 'medication': 1, 'insulin': 2})

# Define features and target
X = df[['age', 'bmi', 'blood_sugar', 'diabetes_type']]
y = df['treatment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Example prediction
example = [[40, 32, 190, 2]]  # Example data: [age, bmi, blood_sugar, diabetes_type]
predicted_treatment = clf.predict(example)
treatment_map = {0: 'lifestyle', 1: 'medication', 2: 'insulin'}
print(f'Predicted treatment: {treatment_map[predicted_treatment[0]]}')
