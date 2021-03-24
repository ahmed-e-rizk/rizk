
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


# import data_set
data = pd.read_csv("D:/project/heart.csv")
data.dropna(inplace=True)  # for removing null values

# splitting to independent and target values
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split to training and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)
# print(x_train, y_train)

st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.transform(x_test)

# learning (fitting the model)
classifier = DecisionTreeClassifier(criterion='gini', random_state=10)
classifier.fit(x_train, y_train)

# fitting the Naive Bayes model
NBclassifier = GaussianNB()
NBclassifier.fit(x_train, y_train)

# fitting the knn model
KNNclassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNNclassifier.fit(x_train, y_train)

# prediction on test set
y_pred = classifier.predict(x_test)
# print(y_pred)

# measuring the accuracy of the model
cm = confusion_matrix(y_test, y_pred)
# print(cm)
# Take input from user
age = float(input("Enter age: "))
sex = float(input("Enter sex: "))
cp = float(input("Enter cp: "))
trestbps = float(input("Enter trestbps: "))
chol = float(input("Enter chol: "))
fbs = float(input("Enter fbs: "))
restecg = float(input("Enter restecg: "))
thalach = float(input("Enter thalach: "))
exang = float(input("Enter exang: "))
oldpeak = float(input("Enter oldpeak: "))
slope = float(input("Enter slope: "))
ca = float(input("Enter ca: "))
thal = float(input("Enter thal: "))


# input must be 2D array
result = classifier.predict(
    [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
print('target ==' + result)
