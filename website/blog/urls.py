from django.urls import path
from . import views

urlpatterns = [
    path('', views.mainpage, name='mainpage'),
    path("process/", views.processPage, name="process"),
    path("process/EDA", views.edaPage, name="eda"),
    path("process/ML", views.mlPage, name="ml"),
    path("process/DL", views.dlPage, name="dl"),
    path("team/", views.teamPage, name="team"),
    path('final/', views.finalPage, name="final"),
    path('final/route/', views.routePage, name="route"),
    path('final/station/', views.stationPage, name="station"),
    path('final/result/', views.resultPage, name="result"),
    path('fun', views.funPage, name="fun"),
]
