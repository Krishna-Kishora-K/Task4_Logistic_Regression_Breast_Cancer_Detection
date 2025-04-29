# Task 4: Logistic Regression Classification (Breast Cancer)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Load dataset
df = pd.read_csv('data.csv')

# Drop unwanted columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Convert target column 'diagnosis' from M/B to 1/0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("\nDataset shape:", df.shape)
print("\nClass distribution:\n", df['diagnosis'].value_counts())
print("\nChecking for nulls:\n", df.isnull().sum())

# Feature and target split
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Print AUC
print(f"ROC-AUC Score: {roc_auc:.2f}")
