import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import joblib


# Load Dataset
df = pd.read_csv("dataset.csv")

print(df.head())


# Input Features
X = df[['Hours', 'Attendance', 'PreviousMarks']]

# Output
y = df['Marks']


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Train Model
model = LinearRegression()

model.fit(X_train, y_train)


# Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)


# Save Model
joblib.dump(model, "model.pkl")

print("Model Saved Successfully")