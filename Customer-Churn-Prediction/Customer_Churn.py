import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv('Telco-Customer-Churn.csv')

print(df.head())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove customerID column
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Fill missing values
df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].mean()
)

# -----------------------------
# EDA
# -----------------------------

sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
plt.show()

sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Gender vs Churn')
plt.show()

plt.hist(df['MonthlyCharges'], bins=20)
plt.title('Monthly Charges')
plt.show()

# -----------------------------
# ENCODING
# -----------------------------

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

# -----------------------------
# FEATURE SELECTION
# -----------------------------

X = df[['tenure', 'Contract', 'MonthlyCharges']]

y = df['Churn']

# -----------------------------
# SPLIT DATA
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# CREATE MODEL
# -----------------------------

model = LogisticRegression()

# -----------------------------
# TRAIN MODEL
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# ACCURACY
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

print('Accuracy:', accuracy)

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(model, 'churn_model.pkl')

print('Model Saved Successfully')