from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split #for splitting the dataset into training set and testing set


import pandas as pd



dataset = pd.read_csv("dataset file.csv")


X_dataset = dataset[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
Y_dataset = dataset.result


# print(Y_dataset, X_dataset)


knn = neighbors.KNeighborsClassifier(n_neighbors= 5,weights ='uniform')

# x_train, x_test, y_train, y_test = train_test_split(X_dataset,Y_dataset, test_size=0.2, random_state=44, shuffle =True)


TP = input("plese enter the Pregnancies ")
TPP = (float(TP))/100

x_train, x_test, y_train, y_test = train_test_split(X_dataset,Y_dataset, test_size=TPP, random_state=44, shuffle =True)

knn.fit(x_train, y_train)


#predictions
prediction = knn.predict(x_test)
#accuracy
accuracy = metrics.accuracy_score(y_test,prediction)




print
print("prediction : ",prediction)
print("accuracy : ",accuracy)


# a = 10
# print('actual_values',Y_dataset[a])
# print('predict_values',knn.predict(x_test)[a])
# # print((x_test)[a])



#inter the number
A = input("plese enter the Pregnancies ")
B = input("plese enter the Glucose ")
C = input("plese enter the BloodPressure ")
D = input("plese enter the SkinThickness ")
E = input("plese enter the Insulin ")
F = input("plese enter the BMI ")
G = input("plese enter the DiabetesPedigreeFunction ")
H = input("plese enter the Age ")

Q = [[A,B,C,D,E,F,G,H]]


print('predict_values',knn.predict(Q))
