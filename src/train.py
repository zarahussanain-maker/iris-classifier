import sklearn
sklearn.datasets.load_iris()
from sklearn.datasets import load_iris
iris = load_iris()
x = iris. data
y= iris.target
print (iris.feature_names,iris.target_names)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predictions:",y_pred[:5])
print("True label:",y_test[:5])
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print (cm)
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
os.makedirs("outputs", exist_ok=True)
fig,ax=plt.subplots()
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.tight_layout()
plt.savefig("outputs\confusion_matrix.png")
plt.close()