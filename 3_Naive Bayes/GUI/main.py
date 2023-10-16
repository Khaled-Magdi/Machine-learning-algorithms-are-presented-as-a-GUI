import tkinter
from tkinter import *
from tkinter import filedialog

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *

#===========================================
# dataset = pd.read_csv('dataset file.csv')
#
# X = dataset.iloc[:,:-1].values
# y = dataset.iloc[:, -1].values
from PIL import ImageTk

dataset =''
datasetName=''
X =''
y =''
def selectFile():
    filePath=filedialog.askopenfilename()
    global dataset
    dataset = pd.read_csv(filePath)
    print(filePath)
    Name =''
    for i in range(0, len(filePath)):
        Name  =Name+ filePath[i]
        if(filePath[i]=='/'):
            Name=''
    print(Name)
    global X
    X= dataset.iloc[:,:-1].values
    global y
    y= dataset.iloc[:, -1].values
    datasetName=Name

    #lable Get file
    ResultAccuracy = Label(window, height=1, width=20, fg="red",background ='#f4faf1',text=datasetName)
    ResultAccuracy.place(x = 411 , y = 194)







def btn_clicked():
    print("Button Clicked")









def percentageGetFn():
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.naive_bayes import GaussianNB
    GaussianNBModel = GaussianNB()
    GaussianNBModel.fit(X_train, y_train)
    GaussianNBModel.score(X_test, y_test)

    print('GaussianNBModel accuracy is : ', GaussianNBModel.score(X_train, y_train))
    y_pred = GaussianNBModel.predict(X_test)
    print('y_pred is : ', y_pred)

    from sklearn.metrics import confusion_matrix
    #
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix is : \n', cm)



    import seaborn as sns
    sns.heatmap(cm, center=True)
    # f=plt.subplots(figsize=(11, 9))
    plt.show()





    #LabelResults
    ResultPredictionTest = Label(window, height=7, width=35,background ='#f2f9f1', fg="red",text=cm)
    ResultPredictionTest.place(x = 655 , y = 207)
    ResultPredictionTest.config(font=('Aria',12))




    ResultAccuracy = Label(window, height=1, width=20, fg="red",background ='#ececec',text=str(int(GaussianNBModel.score(X_test, y_test)*100)) +"%")
    ResultAccuracy.place(x = 265 , y = 379)

    # #LableGraph
    ResultGraph = Label(window, height=20, width=45, fg="red",background ='#ececec',image = plt.show())
    # #f1f8f8
    ResultGraph.place(x = 658 , y = 410)
    ResultGraph.config(justify="center")
    # ResultGraph.config(wraplength =100)
    # # ResultGraph.config(image=z)








def getAllDataFn():
    Q1 = entry2.get()
    Q = Q1.split()
    print(Q)

    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.naive_bayes import GaussianNB
    GaussianNBModel = GaussianNB()
    GaussianNBModel.fit(X_train, y_train)
    GaussianNBModel.score(X_test, y_test)




    # LabelResults
    ResultPredictionTest = Label(window, height=5, width=45,background ='#f3faf1', fg="red", text=str(list(dataset)))
    ResultPredictionTest.place(x = 658 , y = 208)
    ResultPredictionTest.config(wraplength=350)


    ResultPredictionTest1 = Label(window, height=5, width=45,background ='#f2f9f3', fg="red",text= [Q])
    ResultPredictionTest1.place(x = 658 , y = 265)



    ResultAccuracy = Label(window, height=1, width=20, fg="red",background ='#ececec',text=str(int(GaussianNBModel.score(X_test, y_test)*100)) +"%")
    ResultAccuracy.place(x = 265 , y = 379)


    # predictions
    prediction = GaussianNBModel.predict([Q])
    # accuracy
    # accuracy = y_pred
    print('predict_values', GaussianNBModel.predict([Q]))

    ResultPreduct1 = Label(window, height=1, width=20, fg="red",background ='#ececec',text=prediction)
    ResultPreduct1.place(x = 280 , y = 679)








def getDataRowTest():
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TPP, random_state=0)
    # predictions
    # applying the DecisionTreeClassifier algorithm on dataset
    from sklearn.naive_bayes import GaussianNB
    GaussianNBModel = GaussianNB()
    GaussianNBModel.fit(X_train, y_train)
    GaussianNBModel.score(X_test, y_test)

    print('GaussianNBModel accuracy is : ', GaussianNBModel.score(X_train, y_train))
    y_pred = GaussianNBModel.predict(X_test)
    print('y_pred is : ', y_pred)

    a = int(entry1.get()) - 2


    print('first raw is :',str(list(dataset)))
    #LabelResults

    # LabelResults     'hhh \n'+
    ResultPredictionTest = Label(window, height=5, width=45,background ='#f3faf1', fg="red", text=str(list(dataset)))
    ResultPredictionTest.place(x = 658 , y = 208)
    ResultPredictionTest.config(wraplength=350)


    ResultPredictionTest1 = Label(window, height=5, width=45,background ='#f2f9f3', fg="red",text=(X)[a])
    ResultPredictionTest1.place(x = 658 , y = 265)



    ResultAccuracy = Label(window, height=1, width=2, fg="red",background ='#ececec',text=y[a])
    ResultAccuracy.place(x = 565 , y = 606)


    ResultAccuracy1 = Label(window, height=1, width=20, fg="red",background ='#ececec',text=GaussianNBModel.predict(X)[a])
    ResultAccuracy1.place(x = 280 , y = 679)















#GUI-----------------------------------------------









window = Tk()

window.geometry("1024x768")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 768,
    width = 1024,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)



background_img = PhotoImage(file = f"GUI\background.png")
background = canvas.create_image(
    8.0+505, 0.0+384,
    image=background_img)

entry0_img = PhotoImage(file = f"GUI\img_textBox0.png")
entry0_bg = canvas.create_image(
    -167.0+505, -59.0+384,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#ececec",
    highlightthickness = 0)

entry0.place(
    x = -233.0+505, y = -80+384,
    width = 132.0,
    height = 40)





entry1_img = PhotoImage(file = f"GUI\img_textBox1.png")
entry1_bg = canvas.create_image(
    -207.5+505, 233.0+384,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#ececec",
    highlightthickness = 0)

entry1.place(
    x = -233.0+505, y = 212+384,
    width = 51.0,
    height = 40)

getData2 = entry1.get()



entry2_img = PhotoImage(file = f"GUI\img_textBox2.png")
entry2_bg = canvas.create_image(
    -145.5+505, 157.0+384,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#ececec",
    highlightthickness = 0)

entry2.place(
    x = -233.0+505, y = 136+384,
    width = 175.0,
    height = 40)

getData3 = entry2.get()




img0 = PhotoImage(file = f"GUI\img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = percentageGetFn,
    relief = "flat")

b0.place(
    x = -29+505, y = -80+384,
    width = 113,
    height = 42)

img1 = PhotoImage(file = f"GUI\img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = getAllDataFn,
    relief = "flat")

b1.place(
    x = -29+505, y = 136+384,
    width = 113,
    height = 42)

img2 = PhotoImage(file = f"GUI\img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = getDataRowTest,
    relief = "flat")

b2.place(
    x = -151+505, y = 212+384,
    width = 113,
    height = 42)

img3 = PhotoImage(file = f"GUI\img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = selectFile,
    relief = "flat")

b3.place(
    x = -464+505, y = -204+384,
    width = 192,
    height = 42)

window.resizable(False, False)
window.mainloop()
