import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading dataset
dataset = pd.read_csv('heart.csv')


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values
# print(list(dataset))



print(y)


# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# print('X test'+X_test)
# print('y_test'+y_test)
# print('X_train'+X_train)
# print('y_train'+y_train)

#
# applying the GaussianN algorithm on dataset
from sklearn.naive_bayes import GaussianNB
GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)
GaussianNBModel.score(X_test, y_test)

# print('GaussianNBModel Train Score is : ', GaussianNBModel.score(X_train, y_train))
print('GaussianNBModel accuracy is : ', GaussianNBModel.score(X_test, y_test))

y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)

print('y_pred is : ', y_pred)
# print('Prediction Probabilities Value for GaussianNBModel is : ', y_pred_prob[:10])





# Applying MultinomialNB Model
from sklearn.naive_bayes import MultinomialNB


MultinomialNBModel = MultinomialNB(alpha=1.0)
MultinomialNBModel.fit(X_train, y_train)


# Calculating Details
# print('MultinomialNBModel Train Score is : ' , MultinomialNBModel.score(X_train, y_train))
print('MultinomialNBModel accuracy is : ' , MultinomialNBModel.score(X_test, y_test))


# Calculating Prediction
y_pred = MultinomialNBModel.predict(X_test)
print("y_pred : ",y_pred)




# Applying BernoulliNB Model

from sklearn.naive_bayes import BernoulliNB
BernoulliNBModel = BernoulliNB(alpha=1.0,binarize=1)
BernoulliNBModel.fit(X_train, y_train)
#
# # Calculating accuracy
# print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))



 # Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)
print('Predicted Value for BernoulliNBModel is : ' , y_pred)
# # print('Prediction Probabilities Value for BernoulliNBModel is : ' , y_pred_prob[:10])






# entring the values by ur self


input_string = input('Enter elements of a list separated by space ')
user_list = input_string.split()
# print list
print('list: ', [user_list])
print('predict_values is :', GaussianNBModel.predict([user_list]))




# testing accuracy
from sklearn.metrics import confusion_matrix
#
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', cm)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('accuracy is : \n', accuracy)

#
from sklearn.metrics import f1_score

FScore = f1_score(y_test, y_pred)
print('FScore is : \n', FScore)

#
import seaborn as sns

sns.heatmap(cm, center=True)
z=plt.show()
print(z)

# # ----------------------------------------------------------------------------


# chossing the best features
print(GaussianNBModel.feature_importances_)
# using feature_importances_ shows which features are more affecting on data than others