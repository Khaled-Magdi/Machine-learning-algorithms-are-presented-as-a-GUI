import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading dataset
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook')
sns.set_style('white')
#
#
pd.set_option('display.notebook_repr_html',False)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',150)
pd.set_option('display.max_seq_items',None)

# function to draw the data points
def plotData(x, y, size):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(x[pos, 0], x[pos, 1], s=size, c='b', marker='+', linewidths=1)
    plt.scatter(x[neg, 0], x[neg, 1], s=size, c='r', marker='o', linewidths=1)


# Importing the dataset
# dataset = pd.read_csv('../input/dataset/dataset file.csv')
dataset = pd.read_csv("bill_authentication.csv")


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
XDraw = dataset.iloc[:, :-1].values
yDraw = dataset.iloc[:, -1].values
plotData(XDraw, yDraw, 5)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fitting SVM to the Training set
from sklearn.svm import SVC

classifier = SVC(C=100, kernel='rbf', random_state=0)  # best rbf
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred

# Making the Confusion Matrix & testing
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import f1_score

FScore = f1_score(y_test, y_pred)

import seaborn as sns

sns.heatmap(cm, center=True)
plt.show()# using feature_importances_ shows which features are more affecting on data than others