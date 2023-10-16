import pandas as pd

#reading dataset
dataset = pd.read_csv('dataset file.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state = 0)

#applying the DecisionTreeClassifier algorithm on dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
(classifier.fit(X_train, y_train))

print(classifier.score(X_test , y_test))


y_pred = classifier.predict(X_test)
print(y_pred)






 #
a = int(input(" Enter Raw Number "))-2
print('actual_values',y[a])
print('predict_values',classifier.predict(X)[a])
print((X)[a])
#




#
# #
#entring the values by ur self
A = input(" Enter Pregnancies ")
B = input("Enter lucose ")
C = input("Enter BloodPressure ")
D = input("Enter SkinThickness ")
E = input("Enter Insulin ")
F = input("Enter BMI ")
G = input("Enter DiabetesPedigreeFunction ")
H = input("Enter Age ")
Q = [[A,B,C,D,E,F,G,H]]
print('predict_values',classifier.predict(Q))
#
# y_pred = classifier.predict(X_test)
# print(y_pred)


#
#
# testing accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
#
from sklearn.metrics import f1_score
FScore = f1_score(y_test,y_pred)


#----------------------------------------------------------------------------
#chossing the best features

print(classifier.feature_importances_)
#using feature_importances_ shows which features are more affecting on data than others
