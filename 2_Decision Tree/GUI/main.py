import pandas as pd

from tkinter import *

#===========================================


dataset = pd.read_csv('dataset file.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values












# ----------------------GUI

window = Tk()


#Label
LabelFileName =Label(window ,text="File Name Is :")
LabelFileName.grid(row=1,column=0)

LabelEnterTestPercentage =Label(window ,text="Enter percentage of test :")
LabelEnterTestPercentage.grid(row=3,column=0)

LabelAccuracy =Label(window ,text="Accuracy is :")
LabelAccuracy.grid(row=5,column=0)

LabelPredictionTest  =Label(window ,text="Data Table :")
LabelPredictionTest.grid(row=5,column=3)

LabelEnterData =Label(window ,text="Enter the data : ")
LabelEnterData.grid(row=7,column=0)

LabelPrediction  =Label(window ,text="Prediction is : ")
LabelPrediction.grid(row=12,column=0)

#LabelDAat
LabelData1  =Label(window ,text="Pregnancies")
LabelData1.grid(row=8,column=0)

LabelData2  =Label(window ,text="Glucose")
LabelData2.grid(row=8,column=1)

LabelData3  =Label(window ,text="BloodPressure")
LabelData3.grid(row=8,column=2)

LabelData4  =Label(window ,text="SkinThickness")
LabelData4.grid(row=8,column=3)

LabelData5  =Label(window ,text="Insulin")
LabelData5.grid(row=8,column=4)

LabelData6  =Label(window ,text="BMI")
LabelData6.grid(row=8,column=5)

LabelData7  =Label(window ,text="DiabetesPedigreeFunction")
LabelData7.grid(row=8,column=6)

LabelData8  =Label(window ,text="Age")
LabelData8.grid(row=8,column=7)

LabelEnterRaw  =Label(window ,text=" Enter Namber Of Row :")
LabelEnterRaw.grid(row=14,column=0)

LabelRaw  =Label(window ,text=" Aactual value is :")
LabelRaw.grid(row=12,column=4)

#LabelSpaces
LabelSpace  =Label(window ,text=" ")
LabelSpace.grid(row=0,column=0)

LabelSpace  =Label(window ,text=" ")
LabelSpace.grid(row=2,column=0)

LabelSpace  =Label(window ,text=" ")
LabelSpace.grid(row=6,column=0)

LabelSpace  =Label(window ,text=" ")
LabelSpace.grid(row=11,column=0)

LabelSpace  =Label(window ,text=" ")
LabelSpace.grid(row=13,column=0)


LabelSpace  =Label(window ,text="  ")
LabelSpace.grid(row=15,column=0)


LabelSpace  =Label(window ,text="  ")
LabelSpace.grid(row=17,column=0)

LabelSpace  =Label(window ,text="  ")
LabelSpace.grid(row=0,column=8)




#LableBox

ResultFileName = Label(window,height=1,width=20,bg = "white", fg = "red",text="dataset file.csv")
ResultFileName.grid(row=1,column=1)

ResultFileNamePrediction = Label(window,height=1,width=20,bg = "white", fg = "red",text="")
ResultFileNamePrediction.grid(row=12,column=1)

ResultAccuracy = Label(window,height=1,width=20,bg = "white", fg = "red")
ResultAccuracy.grid(row=5,column=1)

ResultPredictionTest = Label(window,height=7,width=70,bg = "white", fg = "red")
ResultPredictionTest.grid(row=5,column=4,columnspan=2)

ResultFileNamePrediction = Label(window, height=1, width=20, bg="white", fg="red")
ResultFileNamePrediction.grid(row=12, column=5)












#inter
getPercentage = StringVar()
p1 = Entry(window, textvariable = getPercentage)
p1.grid(row=3,column=1)

def percentageGetFn():
    TPP = (float(getPercentage.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("prediction : ", y_pred)
    print("classifier score : ", classifier.score(X_test , y_test))
    #LabelResults
    ResultPredictionTest = Label(window, height=7, width=70, bg="white", fg="red",text=y_pred)
    ResultPredictionTest.grid(row=5, column=4, columnspan=2)
    ResultAccuracy = Label(window, height=1, width=20, bg="white", fg="red",text=str(int(classifier.score(X_test , y_test)*100)) +"%")
    ResultAccuracy.grid(row=5, column=1)






getData1 = StringVar()
d1 = Entry(window, textvariable = getData1)
d1.grid(row=9,column=0)


getData2 = StringVar()
d2 = Entry(window, textvariable = getData2)
d2.grid(row=9,column=1)

getData3 = StringVar()
d3 = Entry(window, textvariable = getData3)
d3.grid(row=9,column=2)


getData4 = StringVar()
d4 = Entry(window, textvariable = getData4)
d4.grid(row=9,column=3)


getData5 = StringVar()
d5 = Entry(window, textvariable = getData5)
d5.grid(row=9,column=4)


getData6 = StringVar()
d6 = Entry(window, textvariable = getData6)
d6.grid(row=9,column=5)


getData7 = StringVar()
d7 = Entry(window, textvariable = getData7)
d7.grid(row=9,column=6)


getData8 = StringVar()
d8 = Entry(window, textvariable = getData8)
d8.grid(row=9,column=7)



def getAllDataFn():
    Q = [[getData1.get(),getData2.get(),getData2.get(),getData4.get(),getData5.get(),getData6.get(),getData7.get(),getData8.get()]]


    TPP = (float(getPercentage.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("prediction : ", y_pred)
    print("classifier score : ", classifier.score(X_test , y_test))
    #LabelResults
    ResultPredictionTest = Label(window, height=7, width=70, bg="white", fg="red",text=y_pred)
    ResultPredictionTest.grid(row=5, column=4, columnspan=2)
    ResultAccuracy = Label(window, height=1, width=20, bg="white", fg="red",text=str(int(classifier.score(X_test , y_test)*100)) +"%")
    ResultAccuracy.grid(row=5, column=1)


    # predictions
    prediction = classifier.predict(Q)
    # accuracy
    accuracy = y_pred
    print('predict_values', classifier.predict(Q))
    ResultFileNamePrediction = Label(window, height=1, width=20, bg="white", fg="red", text=prediction)
    ResultFileNamePrediction.grid(row=12, column=1)



#button

buttonDoTest = Button(window,text = "Test", width =12)
buttonDoTest.grid(row=3,column=2)
#
buttonDoTest.config(command =percentageGetFn )








buttonPrediction = Button(window,text = "Prediction", width =12)
buttonPrediction.grid(row=11,column=2)
#
buttonPrediction.config(command =getAllDataFn)







#
#
buttonDoTest = Button(window,text = "Test", width =12)
buttonDoTest.grid(row=14,column=2)
#

getDataRow = StringVar()
R1 = Entry(window, textvariable = getDataRow)
R1.grid(row=14,column=1)



def getDataRowTest():
    TPP = (float(getPercentage.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    a = int(getDataRow.get()) - 2

    print('actual_values', y[a])
    print('predict_values', classifier.predict(X)[a])
    print((X)[a])
    print("prediction : ", y_pred)
    print("classifier score : ", classifier.score(X_test, y_test))
    # LabelResults
    ResultPredictionTest = Label(window, height=7, width=70, bg="white", fg="red", text=(X)[a])
    ResultPredictionTest.grid(row=5, column=4, columnspan=2)
    ResultAccuracy = Label(window, height=1, width=20, bg="white", fg="red",text=str(int(classifier.score(X_test, y_test) * 100)) + "%")
    ResultAccuracy.grid(row=5, column=1)
    ResultFileNamePrediction = Label(window, height=1, width=20, bg="white", fg="red", text=classifier.predict(X)[a])
    ResultFileNamePrediction.grid(row=12, column=1)
    ResultFileNamePrediction = Label(window, height=1, width=20, bg="white", fg="red", text=y[a])
    ResultFileNamePrediction.grid(row=12, column=5)








buttonDoPredictionRow = Button(window,text = "Prediction", width =12)
buttonDoPredictionRow.grid(row=14,column=2)


buttonDoPredictionRow.config(command =getDataRowTest )





classifierFeature = Button(window,text = "classifier Feature", width =12)
classifierFeature.grid(row=3,column=3)

def classifierFeatureDet():
    TPP = (float(getPercentage.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    ResultPredictionTest = Label(window, height=7, width=70, bg="white", fg="red",text=classifier.feature_importances_)
    ResultPredictionTest.grid(row=5, column=4, columnspan=2)
    print(classifier.feature_importances_)



classifierFeature.config(command =classifierFeatureDet )









window.mainloop()



