from tkinter import *
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)





#Import Data_______________________________________________

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



    #handling of data
    Le = LabelEncoder()
    dataset['status_type'] = Le.fit_transform(dataset['status_type'])
    dataset['status_published'] = pd.to_datetime(dataset['status_published'])
    # print(Live.status_published)

    dataset['status_published'] = pd.to_datetime(dataset['status_published']).astype(np.int64)
    # print(Live.status_published)
    # print(Live['status_published'])
    # print(Live.dtypes)

    global X
    X= dataset.iloc[:,1:].values
    global Y
    Y= np.random.rand(7050, 0)
    datasetName=Name

    print(X.shape)
    print(Y.shape)

    #lable Get file
    ResultAccuracy = Label(window, height=1, width=20, fg="red",background ='#e3edfc',text=datasetName)
    ResultAccuracy.place(x = 411 , y = 168)



##Plotting--------------------------------------------------------------




def plotCurve():


    ilist = []
    n = int(entry1.get())
    for i in range(1, n):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        ilist.append(kmeans.inertia_)

    plt.plot(range(1, n), ilist)
    plt.title('Elbow')
    plt.xlabel('clusters')
    plt.ylabel('inertias')
    plt.show()




    labelDraw = Canvas(window)
    labelDraw.place(
        x=655, y=460,
        width=330.0,
        height=280,
    )
    # win = Tk()

    # # setting the title
    # win.title('Plotting in Tkinter')

    # # dimensions of the main window
    # win.geometry("1000x1000")

    # the figure that will contain the plot
    fig = Figure(figsize=(10, 10),
                 dpi=100)

    # list of squares
    # 	y = [i**2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(range(1,n ), ilist)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master=labelDraw)

    # canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   labelDraw)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()





    Itt = int(entry1.get())
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TPP, random_state=44, shuffle=True)

    KMeansModel = KMeans(n_clusters=Itt, init='random', random_state=33, algorithm='auto')
    KMeansModel.fit(X_test)

    Resultn_iter_ = Label(window, height=1, width=7, fg="red", background='#ececec', text=KMeansModel.n_iter_)
    Resultn_iter_.place(x=477, y=422)
    print(KMeansModel.n_iter_)

    ResultAccuracy = Label(window, height=1, width=7, fg="red", background='#ececec', text=KMeansModel.score(X_test))
    ResultAccuracy.place(x=219, y=422)


####Function________________________________________________________________


def percentageGetFn():
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TPP, random_state=44, shuffle=True)



    # predictions

    Itt = int(entry1.get())

    KMeansModel = KMeans (n_clusters= Itt , init='random' ,random_state =33 , algorithm= 'auto')
    print(KMeansModel.fit(X_test))

    print('KMeansModel Train Score is : ', KMeansModel.score(X_train))
    print('KMeansModel Test Score is : ', KMeansModel.score(X_test))
    print('KMeansModel centers are : ', KMeansModel.cluster_centers_)
    print('KMeansModel labels are : ', KMeansModel.labels_)
    print('KMeansModel intertia is : ', KMeansModel.inertia_)
    print('KMeansModel No. of iteration is : ', KMeansModel.n_iter_)
    # print('----------------------------------------------------')

    rX =KMeansModel.score(X_test)
    # rX1 = KMeansModel.score(X_test) *-1
    # print(rX.dtypes)
    # rX2 =round(  , 3)
    # print(rX2 )

    labelDraw = Canvas(window)
    labelDraw.place(
        x=655, y=460,
        width=330.0,
        height=280,
    )
    # win = Tk()

    # # setting the title
    # win.title('Plotting in Tkinter')

    # # dimensions of the main window
    # win.geometry("1000x1000")

    # the figure that will contain the plot
    fig = Figure(figsize=(10, 10),
                 dpi=100)

    # list of squares
    # 	y = [i**2 for i in range(101)]
    kmeans = KMeans(n_clusters=Itt, init='k-means++', random_state=33, algorithm='auto')
    y_kmeans = kmeans.fit_predict(X)
    # adding the subplot
    plot1 = fig.add_subplot(111)
    plot1.cla()
    plot1.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='r')
    plot1.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='b')
    plot1.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='g')
    plot1.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c='c')
    plot1.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='m')
    plot1.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='y')
    plot1.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='pink')
    plot1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='y')
    # plot1.title('Clusters of customers')
    # plot1.xlabel('Annual Income (k$)')
    # plot1.ylabel('Spending Score (1-100)')
    # plot1.legend()
    plot1.plot()

    # plotting the graph
    # plot1.plot(range(1,n ), ilist)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master=labelDraw)

    # canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   labelDraw)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()


    # #LabelResults
    ResultPredictionTest = Label(window, height=18, width=64,background ='#e8f0f5', fg="red",text=KMeansModel.cluster_centers_)
    ResultPredictionTest.place(x = 655 , y = 227)
    ResultPredictionTest.config(font=('Aria',6))




    ResultAccuracy = Label(window, height=1, width=7, fg="red",background ='#ececec',text=rX)
    ResultAccuracy.place(x=219, y=422)
    #
    Resultn_iter_ = Label(window, height=1, width=7, fg="red", background='#ececec', text=KMeansModel.n_iter_)
    Resultn_iter_.place(x=477, y=422)

    # #LableGraph




def getNumberOfRaw():


    # print(entry2.get())
    Itt = int(entry1.get())
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TPP, random_state=44, shuffle=True)

    KMeansModel = KMeans(n_clusters=Itt, init='random', random_state=33, algorithm='auto')
    KMeansModel.fit(X_test)
    Y_pred = KMeansModel.predict(X_test)
    print('Predicted Value for KMeansModel is : ', Y_pred[int(entry2.get())])



    ResultAccuracy = Label(window, height=1, width=7, fg="red",background ='#ececec',text=Y_pred[int(entry2.get())])
    ResultAccuracy.place(x=375, y=683)

    # #LableGraph



def getRandomArray():


    # print(entry3.get())
    Itt = int(entry1.get())
    TPP = (float(entry0.get())) / 100
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TPP, random_state=44, shuffle=True)

    KMeansModel = KMeans(n_clusters=Itt, init='random', random_state=33, algorithm='auto')
    KMeansModel.fit(X_test)
    Y_pred = KMeansModel.predict(X_test)
    print('Predicted Value for KMeansModel is : ', Y_pred[:int(entry3.get())])



    ResultPredictionTest = Label(window, height=10, width=36, background='#e9f0f2', fg="red",text=Y_pred[:int(entry3.get())])
    ResultPredictionTest.place(x=655, y=227)
    ResultPredictionTest.config(font=('Aria', 12))

    # #LableGraph



def btn_clicked():
    print("Button Clicked")
##Window________________________________________________________________





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






#Entrys________________________________________________________________
entry0_img = PhotoImage(file = f"GUI\img_textBox0.png")
entry0_bg = canvas.create_image(
    -234.5+505, -71.0+384,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#dde9ed",
    highlightthickness = 0)

entry0.place(
    x = -248.0+505, y = -92+384,
    width = 27.0,
    height = 40)

entry1_img = PhotoImage(file = f"GUI\img_textBox1.png")
entry1_bg = canvas.create_image(
    46.5+505, -71.0+384,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#dde9ed",
    highlightthickness = 0)

entry1.place(
    x = 33.0+505, y = -92+384,
    width = 27.0,
    height = 40)

entry2_img = PhotoImage(file = f"GUI\img_textBox2.png")
entry2_bg = canvas.create_image(
    -266.5+505, 198.0+384,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#dde9ed",
    highlightthickness = 0)

entry2.place(
    x = -280.0+505, y = 177+384,
    width = 27.0,
    height = 40)

entry3_img = PhotoImage(file = f"GUI\img_textBox3.png")
entry3_bg = canvas.create_image(
    46.5+505, 198.0+384,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#dde9ed",
    highlightthickness = 0)

entry3.place(
    x = 33.0+505, y = 177+384,
    width = 27.0,
    height = 40)




#Button________________________________________________________________

img0 = PhotoImage(file = f"GUI\img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = percentageGetFn,
    relief = "flat")

b0.place(
    x = -377+505, y = -34+384,
    width = 180,
    height = 42)

img1 = PhotoImage(file = f"GUI\img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = getNumberOfRaw,
    relief = "flat")

b1.place(
    x = -362+505, y = 235+384,
    width = 137,
    height = 42)

img2 = PhotoImage(file = f"GUI\img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = getRandomArray,
    relief = "flat")

b2.place(
    x = -158+505, y = 235+384,
    width = 137,
    height = 42)

img3 = PhotoImage(file = f"GUI\img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = plotCurve,
    relief = "flat")

b3.place(
    x = -178+505, y = -34+384,
    width = 180,
    height = 42)

img4 = PhotoImage(file = f"GUI\img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = selectFile,
    relief = "flat")

b4.place(
    x = -464+505, y = -231+384,
    width = 192,
    height = 42)

window.resizable(False, False)
window.mainloop()
