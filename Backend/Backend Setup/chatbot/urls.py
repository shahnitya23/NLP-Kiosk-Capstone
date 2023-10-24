from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main, name = "Kiosk"),
    path('training/', views.training, name = "Training Model"),
]