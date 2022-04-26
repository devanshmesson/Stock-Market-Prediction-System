from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn import linear_model

root = Tk()
root.title("Stock Market Prediction and Analysis")
root.configure(background='antique white')

label1 = Label(root, text="Stock Prediction and Analysis", fg="black", font="Helvetica 20 bold italic underline",
               bg='antique white')
label1.place(x=440, y=0)

button1 = Button(root, text="My Tab", width=15, fg="red", font="times 13 bold", bg='white')
button1.place(x=120, y=20)

# -------------------------------// LOGO //----------------------------------------------#

c1 = Canvas(root, width=200, height=80, bg='antique white')
photo1 = '''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'''
photo1 = PhotoImage(data=photo1)
image = c1.create_image(0, 42, image=photo1, anchor=W)
c1.place(x=1000, y=10)

# ----------------------------------------------------------------------#

c2 = Canvas(root, width=45, height=45, bg='antique white')
photo2 = '''iVBORw0KGgoAAAANSUhEUgAAACYAAAAmCAYAAACoPemuAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QA/wD/AP+gvaeTAAAAB3RJTUUH5AIBAAcPXINMZwAABHBJREFUWMPV2FuIlFUcAPDf+b7Z2d0WK1qtsKthumtBUK6WhkZFTBC2XYyyXgp7qSjMgigNip4KkqIoigiiVy89GJjQDbKrVnTbalXSbqzrWkpK7c6cHr6Z3MvMuuqM0P9tvjnfd37zP2e+8z8nmGis6iYMEvMnYTYuwxxMx2S0llseQD968QnexWfCPwNiE8vWTai7MCEQgTANN6Ab52PSBO6P2IevsQ6ridsRDwWs/eBnr2coEkzG7biznJ1D/5jayF68jFdF/XKBe9ccBmxVN6FETOfhCSxAeoSg0VHE+1ghFDeJSdXhHdvZqm7ElHQJXsQFSOqEUn7WNFxF0kf8VqEz2tAzDiybTynJUjyNU+oIGh0n4HLCAL5Q6BiBOwhb1V35uKSMOrGBqEq04FLCDtKvFGaq4DLYU4tIEsT5eKExmYrySU4uJIqxZNj0bkEX8RPCTlfOYOP35bmTSxGn4HGcUX9T1JrmrZx7k8cuuUVbrpkYh7c4I+s7TskspMOG8G4sdeSvg9qoXN6KuYs9OPs686Z2aM3lbfq1x2BxiPBfd2cR+kk2KcyUKnQiniObV5MbhXrgomvl0yZpSHSdOr0aLmS4+CZhT6owSzlTNzcaVYkKri3X7IPfegyWhioD1Y5fCJsSYrtsmTkmqEo0JTnzpnY4sbktWxMORjexPZEtyOcdSxRs6dvqrrdf8su+3cPnmbJldoKFspfdMUUt3fi8z/u2jkYpWxYm6BqnJ2kIE1uPDhu1jVDzyV2JrGKo2tFxuWYPd93ots6F4+Pqi4LpOUypjspbOXex5Rd12/vPfvD6d+8pNR4FUxIHK8+RqItvcv+F12pKUu0tkzy94I6xmWsMClpzo6+05VusnLPYsgsXyacHv25vzXD/Za5xKJCT1ehtWQZKzj7+ZItnzB+BGoOLrO790MNzbmwICgdShc47VUqcEOw6sNfP+/otOH2WSfnWMXcc19Rs/tQOF586w62dC7Xk8vVGwa+pQuc1OGf41e8Gfrb9z76auLamFjNPOk1TUr3a3tK37WhQ8FmCT8dcDsGa3o/c8/bLfvtr4LCemGXquaNBwaepQmeKRbKCbQTuUJmrjjqqTMGfeCohbsY3VZscRubqhJJZ4uaEZLdsM1o9JoCrI0pmSXanCh3wO66W1UNVcbWGtc6oH/AI9pRh6R5iK65Qq7SugqvDv294lPAkyXpKUht6lLP2o6w2m1bz1jLup727TMq3WvbuK/VCwTtYQdxv2TrZ632oSC7dhUdxrvF2SiFY2/uRt3Z8ad/f++uF2lnue5ehIir7yo3fl7OW7iT24XKjXx/DIuLv4mC1Iu9I4g8sF9L1lFj+xjAY2Q64MBO+LW/bLx0PV6dd3h94iPgapRFHUyPXlGy+ReIX2bZdl3qV3WNjJ5ZTeo1QHH3iM3ax29BDoTMK8SvCx7J19Ez1O/Ep4j3cJRTfJKl6iFd7PJ65TrlcbczBHf0S3Le2auP/4VFnVeAQmo7scNjgALkJHw7/Cwqr+jBV2zDuAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDIwLTAyLTAxVDAwOjA3OjE1KzAwOjAwqg7F0gAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyMC0wMi0wMVQwMDowNzoxNSswMDowMNtTfW4AAAAZdEVYdFNvZnR3YXJlAEFkb2JlIEltYWdlUmVhZHlxyWU8AAAAAElFTkSuQmCC'''
photo2 = PhotoImage(data=photo2)
image = c2.create_image(5, 25, image=photo2, anchor=W)
c2.place(x=1000, y=200)

c68 = Frame(root, width=45, height=45, bg='antique white')
c68.place(x=1000, y=200)

# -----------------------------------------------------
label1 = Label(root, text="Enter the Stock ticker symbol listed for the company :", bg='antique white', fg="dark green",
               font="times 15")
label1.place(x=30, y=60)
v = Entry(root, width=20, borderwidth=2, bg='antique white')
v.place(x=460, y=65)

button1 = Button(root, text="Solve", width=6, fg="red", font="Arial 13 bold", bg='antique white',
                 command=lambda: solve(), )
button1.place(x=600, y=60)

Str = str(v.get())


# ------------------------------- // Backend //----------------------------------------------#
def SentiAnal(headlines):
    nwin = Toplevel()
    nwin.configure(background='azure')

    scrollbar = Scrollbar(nwin)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(nwin, font='Helvetica 15', fg='red', bg='black')
    listbox.pack(side="left", fill="both", expand=True)

    for i in range(len(headlines)):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(headlines[i])
        listbox.insert(END, str(i + 1) + ') ' + headlines[i])
        s1 = "Overall sentiment dictionary is : " + str(sentiment_dict) + '\n'
        listbox.insert(END, s1)
        s2 = "Sentence was rated as " + str(sentiment_dict['neg'] * 100) + "% Negative" + '\n'
        listbox.insert(END, s2)
        s3 = "Sentence was rated as " + str(sentiment_dict['neu'] * 100) + "% Neutral" + '\n'
        listbox.insert(END, s3)
        s4 = "Sentence was rated as " + str(sentiment_dict['pos'] * 100) + "% Positive" + '\n' + '\n'
        listbox.insert(END, s4)
        s5 = "Sentence Overall Rated As "

        if sentiment_dict['compound'] >= 0.05:
            s6 = "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            s6 = "Negative"

        else:
            s6 = "Neutral"

        listbox.insert(END, s5 + s6 + '\n')

        listbox.insert(END, '\n')

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    nwin.state('zoomed')

#step-2
def SentimentAnalysis(headlines):
    sum = 0
    for i in range(len(headlines)):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(headlines[i])

        if sentiment_dict['compound'] >= 0.05:
            x = 1

        elif sentiment_dict['compound'] <= - 0.05:
            x = -1

        else:
            x = 0

        sum += x

    return sum


def NewsWindow(headlines):
    nwin = Toplevel()
    nwin.configure(bg='yellow')

    scrollbar = Scrollbar(nwin)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(nwin, font='Helvetica 15')
    listbox.pack(side="left", fill="both", expand=True)

    for i in range(len(headlines)):
        listbox.insert(END, str(i + 1) + ') ' + headlines[i])

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    nwin.state('zoomed')


def PredictionGraph():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import LSTM, Dense, Dropout
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.dates as mdates
    from sklearn import linear_model

    sName = str(v.get())
    df_final = pd.read_csv("D:/College/Hackathon/AAPL/" + sName + '.csv', na_values=['null'], index_col='Date', parse_dates=True,
                           infer_datetime_format=True)
    # if(sName=='MSFT')df_final = pd.read_csv("D:/MSFT.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
    X = df_final.drop(['Adj Close'], axis=1)
    X = X.drop(['Close'], axis=1)

    test = df_final
    # Target column
    target_adj_close = pd.DataFrame(test['Adj Close'])

    feature_columns = ['Open', 'High', 'Low', 'Volume']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data,
                                            index=test.index)
    Last_day_data = list()
    for i in feature_minmax_transform:
        Last_day_data.append(feature_minmax_transform[i][len(feature_minmax_transform) - 1])

    Last_day = list()
    Last_day.append(Last_day_data)
    Last_day = np.array(Last_day)

    Prediction_date = df_final.index[len(df_final.index) - 1] + datetime.timedelta(days=1)
    Prediction_date = Prediction_date.strftime('%Y-%m-%d')




    target_adj_close = target_adj_close.shift(-1)
    target_adj_close = target_adj_close[:-1]
    feature_minmax_transform = feature_minmax_transform[:-1]

    s_split = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in s_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (
                    len(train_index) + len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(), target_adj_close[len(train_index): (
                    len(train_index) + len(test_index))].values.ravel()

    Dates = X_test.index
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    from keras.models import Sequential
    from keras.layers import Dense
    import keras.backend as K
    from keras.callbacks import EarlyStopping
    #from keras.optimizers import Adam
    from tensorflow.keras.optimizers import Adam
    from keras.models import load_model
    from keras.layers import LSTM
    K.clear_session()
    model_lstm = Sequential()
    model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False,
                                        callbacks=[early_stop])

    y_pred_test_lstm = model_lstm.predict(X_tst_t)
    y_train_pred_lstm = model_lstm.predict(X_tr_t)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
    r2_train = r2_score(y_train, y_train_pred_lstm)

    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
    r2_test = r2_score(y_test, y_pred_test_lstm)
    score_lstm = model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

    fig = Figure(figsize=(7,7),dpi=70)
    plt = fig.add_subplot(111)

    y_pred_test_LSTM = model_lstm.predict(X_tst_t)
    plt.plot(Dates,y_test, label='Actual Value')
    plt.plot(Dates,y_pred_test_LSTM, label='LSTM (Predicted)')
    for tick in plt.get_xticklabels():
        tick.set_rotation(90)

    plt.set_xlabel('Date')
    plt.set_ylabel('Stock Price')
    plt.legend()



    canvas8 = Canvas(root, width=6, height=4)
    canvas8 = FigureCanvasTkAgg(fig, master=root)
    canvas8.get_tk_widget().place(x=30, y=260)
    canvas8.draw()

    Last_day = Last_day.reshape(1, 1, Last_day.shape[1])
    Last_day_price = model_lstm.predict(Last_day)

    label1 = Label(root, text="Stock Price Prediction:",
                   fg="black", font="times 15 bold",
                   bg='antique white')
    label1.place(x=950, y=400)

    Print_text = "Stock Price on " + Prediction_date
    addon = f" for {sName} is predicted to be Rs. {Last_day_price[0][0]:.2f}"
    Final = Print_text + addon
    label1 = Label(root, text=Final,
                   fg="black", font="times 13 bold",
                   bg='antique white')
    label1.place(x=900, y=450)


#Starting from here:1
def solve():
    sName = str(v.get())
    url=''
    if (sName == ''):
        cnvs2 = Canvas(root, width=5, height=28, bg='antique white')
        cnvs2.place(x=50, y=100)
        label = Label(cnvs2, text="Please enter a company's ticker symbol (like AAPL)", fg='red', bg='yellow', font=3)
        label.place(x=0, y=0)

    elif (sName == "AAPL"):
        url = ('https://newsapi.org/v2/everything?'
               'q=APPLE&'
               'pageSize=50&'
               'from=2022-04-14&'
               'language = en&'
               'to=2022-04-24&'
               'sortBy = popularity&'
               'apiKey=d0708e0c9220410984f0081e01b998a6')

    elif (sName == 'MSFT'):
        url = ('https://newsapi.org/v2/everything?'
               'q=Microsoft&'
               'pageSize=50&'
               'from=2022-04-14&'
               'language = en&'
               'to=2022-04-24&'
               'sortBy = popularity&'
               'apiKey=d0708e0c9220410984f0081e01b998a6')
    elif (sName == 'GOOGL'):
        url = ('https://newsapi.org/v2/everything?'
               'q=Google&'
               'pageSize=50&'
               'from=2022-04-14&'
               'language = en&'
               'to=2022-04-24&'
               'sortBy = popularity&'
               'apiKey=d0708e0c9220410984f0081e01b998a6')

    open_url = requests.get(url).json()

    news = open_url['articles']

    headlines = []

    for ar in news:
        headlines.append(ar['title'])

    s1 = ''

    temp = 130

    label = Label(root, text='Headlines', font='times 15 bold italic underline', fg='black', bg='antique white')
    label.place(x=8, y=100)

    for i in range(1, 6, 1):
        label = Label(root, text=str(i) + ') ' + headlines[i], font='Helvetica 11', fg='black', bg='antique white')
        label.place(x=8, y=temp)
        temp += 25

    senti_fin = SentimentAnalysis(headlines)

    if (senti_fin >= 10):
        label1 = Label(root,
                       text="   We found this company to be on the positive side   " + '\n' + "(Based on various trusted news sources)",
                       bg='yellow', fg="red", font="times 15")
        label1.place(x=810, y=130)
        c68.destroy()

    elif (senti_fin <= -1):
        label1 = Label(root,
                       text="   We found this company to be on the negative side   " + '\n' + "(Based on various trusted news sources)",
                       bg='black', fg="white", font="times 15")
        label1.place(x=810, y=130)
    else:
        label1 = Label(root,
                       text="   We found this company to be on the neutral side    " + '\n' + "(Based on various trusted news sources)",
                       bg='antique white', fg="black", font="times 15")
        label1.place(x=810, y=130)

    btheadlines = Button(root, text='Show all Headlines', command=lambda: NewsWindow(headlines), bg='white',
                         fg='red').place(x=140, y=100)
    btSentiAnal = Button(root, text='Show detailed Sentimental Analysis', font='Helvetica 10', fg='red', bg='white',
                         command=lambda: SentiAnal(headlines)).place(x=930, y=250)
    btgraph = Button(root, text='Show Prediction Graph', font='Helvetica 10', fg='red', bg='white',
                     command=lambda: PredictionGraph()).place(x=100, y=280)


# -----------------------------------------------------------------------------

root.state('zoomed')
root.mainloop()
