"""
URL configuration for proyekjamur project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

def fake_login(request):
    return redirect('/')

def fake_logout(request):
    return redirect('/')

def fake_register(request):
    return redirect('/')

# home stub supaya template yang memanggil {% url 'home' %} tidak error
def home_view(request):
    return redirect('/')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', fake_login, name='login'),
    path('logout/', fake_logout, name='logout'),
    path('register/', fake_register, name='register'),
    path('home/', home_view, name='home'),
    # path('akun/', include('akun.urls')),
    path('', include('prediksi.urls')),
]
