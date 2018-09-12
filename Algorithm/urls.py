from django.urls import path

from Algorithm.views import VideoList, DeleteVideo

urlpatterns = [
    path('list_video', VideoList.as_view(), name='list_video'),
    path('video/(?P<pk>\d+)/delete/', DeleteVideo.as_view(), name='delete_video')
]
