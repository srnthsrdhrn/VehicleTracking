from django.urls import path

from Algorithm.views import VideoList, DeleteVideo, VideoOutput, BarChartOutput

urlpatterns = [
    path('list_video', VideoList.as_view(), name='list_video'),
    path('video/(?P<pk>\d+)/delete/', DeleteVideo.as_view(), name='delete_video'),
    path('output_video_graph/(?P<pk>\d+)/(?P<is_move_avg>\d+)', VideoOutput, name='output_process'),
    path('graph_output_video/(?P<pk>\d+)', BarChartOutput, name='graph_output_video')
]
