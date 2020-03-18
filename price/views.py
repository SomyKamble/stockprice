from django.shortcuts import render
from django.http import HttpRequest ,HttpResponseRedirect
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
# Create your views here.
from sklearn.tree import DecisionTreeRegressor


def home(request):

    g=1


    return render(request,'home.html',{'g':g})

def predict(request):


    try:


        sam=request.POST.get('stock')
        #start = dt.datetime(2018, 2, 11)
        end = dt.datetime.today()
        stock = sam
        df = yf.download(stock, start="2018-02-11", end=end)

        def yr(s):
            s = s.split("-")
            return int(s[0])

        def mnth(s):
            s = s.split("-")
            return int(s[1])

        def day(s):
            s = s.split("-")
            return int(s[2])

        df = df.reset_index()
        year1 = df['Date'].astype('str')
        month1 = df['Date'].astype('str')
        day1 = df['Date'].astype('str')
        year1 = year1.apply(yr)
        month1 = month1.apply(mnth)
        day1 = day1.apply(day)
        # adding the columns into the dataset1
        df["year"] = year1
        df["month"] = month1
        df["day"] = day1
        df = df[['year', 'month', 'day', 'Open', 'High', 'Low', 'Close', 'Volume']]
        y = df.iloc[:, 6]
        x = df.drop(columns=['Close', 'Volume'])
        # Dividing the test and train sequentially

        l = len(x)
        s = round(l * 0.8)

        # Dividing the test and train sequentially

        x_train = x.iloc[0:s].values
        x_test = x.iloc[s + 1:l].values
        y_train = y.iloc[0:s].values
        y_test = y.iloc[s + 1:l].values

        model_dtree = DecisionTreeRegressor().fit(x_train, y_train)
        pred = model_dtree.predict(x_test)

        plt.figure(figsize=(20, 10))
        plt.plot(pred, color='c', label='predicted')
        plt.title("Prediction Vs actual Values using Decision tree")

        plt.plot(y_test, color='g', label='actual')
        plt.legend()
        plt.savefig('st/ram.png')
        plt.close()

        mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50,
                           max_iter=150, shuffle=True, random_state=1,
                           activation='relu')
        mlp.fit(x_train, y_train)

        pred_mlp = mlp.predict(x_test)
        print(pred_mlp)
        plt.figure(figsize=(15, 9))
        plt.plot(y_test, label="actual")
        plt.plot(pred_mlp, label="predicted")
        plt.title("Prediction Vs actual Values Using Artifical Neural Network")
        plt.legend()
        plt.savefig('st/sam.png')
        plt.close()

        model_lm = LinearRegression().fit(x_train, y_train)
        pred2 = model_lm.predict(x_test)
        print(pred2)
        plt.figure(figsize=(20, 10))
        plt.plot(pred2, color='c', label='predicted')
        plt.title("Prediction Vs actual Values using Linear Regression")
        plt.plot(y_test, color='g', label='actual')
        plt.legend()
        plt.savefig('st/tam.png')
        plt.close()

    except:
        g=0
        return render(request,'home.html',{'g':g})

    return render(request, 'result.html', {'sam': sam})


