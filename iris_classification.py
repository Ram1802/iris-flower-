# Iris Flower Classification Project

# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the Iris dataset
iris = load_iris()

# Convert dataset to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map numeric target to flower names
df['species_name'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

# Display first 5 rows
print("First 5 rows of dataset:\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['species_name'].value_counts())

# Step 3: Data Preprocessing
# Features and target
X = df[iris.feature_names]
y = df['species']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train classification model (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Display model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Predict flower species based on user input
def predict_flower():
    print("\n--- Predict Iris Flower Species ---")

    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    user_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names
    )

    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)

    print("\nPredicted Iris Flower Species:", iris.target_names[prediction[0]].capitalize())

predict_flower()

# Prepare input data
user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_data_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_data_scaled)
predicted_species = iris.target_names[prediction[0]]

print("\nPredicted Iris Flower Species:", predicted_species.capitalize())
