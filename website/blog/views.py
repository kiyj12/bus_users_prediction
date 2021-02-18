from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import numpy as np #데이터 처리
import pandas as pd #데이터 처리

import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import joblib

# Create your views here.
def mainpage(request):
    return render(request, 'blog/mainpage.html', {})

def processPage(request):
    return render(request, 'blog/process.html', {})

def edaPage(request):
    return render(request, 'blog/eda.html', {})
def mlPage(request):
    return render(request, 'blog/ml.html', {})
def dlPage(request):
    return render(request, 'blog/dl.html', {})

def teamPage(request):
    return render(request, 'blog/team.html', {})

def funPage(request):
    return render(request, 'blog/fun_fact.html', {})



@csrf_exempt
def finalPage(request):
    return render(request, 'blog/final.html',{})

@csrf_exempt
def routePage(request):

    selected_date = request.POST["selectedDate"]
    df = pd.read_csv("test_web.csv")
    df = df[df["date"]==selected_date]
    route_ids = list(df["bus_route_id"].unique())
    route_ids.sort()
    return render(request, 'blog/route.html',{"dateInfo":selected_date,"routeList":route_ids})

@csrf_exempt
def stationPage(request):
    selected_date = request.POST["selectedDate"]
    selected_route = request.POST["selectedRoute"]

    df = pd.read_csv("test_web.csv")
    df = df[df["date"]==selected_date]
    station_names = list(df[df["bus_route_id"]==int(selected_route)]["station_name"].unique())
    station_names.sort()

    return render(request, 'blog/station.html',{"dateInfo":selected_date,"routeInfo":selected_route, "stationList": station_names})

@csrf_exempt
def resultPage(request):
    selected_date = request.POST["selectedDate"]
    selected_route = request.POST["selectedRoute"]
    selected_station = request.POST["selectedStation"]

    df = pd.read_csv("test_web.csv")

    df = df[df["date"] == str(selected_date)]
    df = df[df["bus_route_id"] == int(selected_route)]
    df = df[df["station_name"]== selected_station]

    del df["date"]
    del df["station_code"]
    del df["station_name"]
    del df["day"]

    model = joblib.load('lgbm.pkl')

    pred = np.expm1(model.predict(df))[0]
    pred = int(pred)

    return render(request, 'blog/result.html',{"dateInfo":selected_date,"routeInfo":selected_route, "stationInfo": selected_station, "prediction":pred})
