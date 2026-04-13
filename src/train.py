# All imports at the top
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load data
iris = load_iris()
x = iris.data
y = iris.target
print(iris.feature_names, iris.target_names)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print("Predictions:", y_pred[:5])
print("True label:", y_test[:5])

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save all outputs
os.makedirs("outputs", exist_ok=True)

# Save trained model
joblib.dump(model, "outputs/iris_model.pkl")
print("Model saved to outputs/iris_model.pkl")

# Save confusion matrix image
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to outputs/confusion_matrix.png")