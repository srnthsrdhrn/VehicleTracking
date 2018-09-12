"""SmartCity URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
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
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from Algorithm.views import draw_canvas, get_video_input, start_processing, stop_processes, landing_page
from SmartCity import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('draw/(?P<pk>\d+)/', draw_canvas, name='draw_canvas'),
    path('get_video_input', get_video_input, name='get_video_input'),
    path('stop_process/(?P<pk>\d+)/', stop_processes, name="stop_process"),
    path('start_processing/(?P<pk>\d+)/', start_processing, name='start_processing'),
    path('', landing_page, name='landing_page'),
    path('algorithm/',include('Algorithm.urls')),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
