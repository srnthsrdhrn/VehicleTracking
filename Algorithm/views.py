import json
import time
from queue import PriorityQueue, Queue
from threading import Thread

import cv2
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect
# Create your views here.
from django.urls import reverse
from django.views.generic import ListView, DeleteView
from pytz import timezone

from Algorithm.forms import VideoProcessForm
from Algorithm.models import Videos, VideoLog
from SmartCity import settings


def landing_page(request):
    return render(request, 'theme/admin/base.html')


def get_video_input(request):
    form = VideoProcessForm()
    if request.method == 'POST':
        form = VideoProcessForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            return redirect('draw_canvas', obj.id)
    return render(request, "algorithm/get_video_input.html", {"form": form})


def conversion(initx, inity, endx, endy, frame_width, frame_height):
    """
    Utility Method to convert the co ords in Tkinter to the co ords in image
    :return: Parsed Line Co ordinates
    """
    start_x = round(initx / 900 * frame_width)
    end_x = round(endx / 900 * frame_width)
    start_y = round(inity / 600 * frame_height)
    end_y = round(endy / 600 * frame_height)
    line_coordinate = [start_x, start_y, end_x, end_y]
    return line_coordinate


def draw_canvas(request, pk):
    video = Videos.objects.get(id=pk)
    path = None
    width = 0
    height = 0
    if request.method == "POST":
        data = request.POST
        coords = data.get("value")
        init, end = coords.split("/")
        ix, iy = init.split("|")
        ex, ey = end.split("|")
        if video.file:
            cap = cv2.VideoCapture(video.file.path)
        else:
            if video.ip_link == '0':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(video.ip_link)
        ret, frame = cap.read()
        width = frame.shape[1]
        height = frame.shape[0]
        video.line_coord_init_x, video.line_coord_init_y, video.line_coord_end_x, video.line_coord_end_y = conversion(
            int(ix), int(iy), int(ex), int(ey), width, height)
        video.save()
        cap.release()
        return redirect("list_video")
    else:
        if video.file:
            cap = cv2.VideoCapture(video.file.path)
            ret, frame = cap.read()
            path = "media/tmp/{}.jpg".format(video.file.name.split(".")[0])
            cv2.imwrite(path, frame)
            cap.release()
        elif video.ip_link:
            if video.ip_link == '0':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(video.ip_link)
            ret, frame = cap.read()
            path = "media/tmp/{}.jpg".format("test")
            cv2.imwrite(path, frame)
    path = "http://" + request.get_host() + "/" + path
    line_coordinates = None
    if video.line_coord_init_x:
        line_coordinates = [video.line_coord_init_x, video.line_coord_init_y, video.line_coord_end_x,
                            video.line_coord_end_y]
    return render(request, "algorithm/template.html",
                  {"image_url": path, "line_coordinates": line_coordinates, 'video_id': pk, 'width': width,
                   'height': height})


def start_processing(request, pk):
    t = Thread(target=initiate_process, args=(pk,))
    t.daemon = True
    t.start()
    messages.success(request, "Processing Started")
    return redirect("list_video")


def initiate_process(pk):
    from traffic_counter import DeepSenseTrafficManagement
    video = Videos.objects.get(id=pk)
    video.processed = True
    video.save()
    line_coordinates = [video.line_coord_init_x, video.line_coord_init_y, video.line_coord_end_x,
                        video.line_coord_end_y]
    path = video.file.path if video.file else video.ip_link
    file = video.file.name if video.file else "IPCAM"
    resultQueue = PriorityQueue()
    bufferQueue = Queue()
    settings.resultQueueDict[pk] = resultQueue
    settings.bufferQueueDict[pk] = bufferQueue
    DeepSenseTrafficManagement(line_coordinates, file, pk, path)


def stop_processes(request, pk):
    video = Videos.objects.get(id=pk)
    video.processed = True
    video.save()
    time.sleep(3)
    video.processed = False
    video.save()
    messages.success(request, "Video Processing Stopped")
    return redirect("list_video")


class VideoList(ListView):
    model = Videos
    template_name = 'algorithm/list_Video.html'


class DeleteVideo(DeleteView):
    model = Videos
    template_name = 'algorithm/delete.html'

    def get_success_url(self):
        return reverse('list_video')


def VideoLogAPI(request):
    video_id = request.GET.get("video_id")
    logs = VideoLog.objects.filter(video_id=video_id).order_by('-created_at')
    is_moving_avg = request.GET.get("is_moving_avg")
    if is_moving_avg and int(is_moving_avg) == 1:
        logs = logs.filter(moving_avg__isnull=False)
    else:
        logs = logs.filter(moving_avg__isnull=True)
    if logs.exists():
        logs = logs[:10]
    car_inflow = []
    car_outflow = []
    bike_inflow = []
    bike_outflow = []
    truck_inflow = []
    truck_outflow = []
    time = []
    for log in logs:
        if is_moving_avg and int(is_moving_avg) == 1:
            temp = json.loads(log.moving_avg)
            car_inflow.append(temp[0])
            car_outflow.append(temp[3])
            bike_inflow.append(temp[1])
            bike_outflow.append(temp[4])
            truck_inflow.append(temp[2])
            truck_outflow.append(temp[5])
        else:
            temp = json.loads(log.data)
            car_inflow.append(temp[0])
            car_outflow.append(temp[4])
            bike_inflow.append(temp[1])
            bike_outflow.append(temp[5])
            truck_inflow.append(temp[2])
            truck_outflow.append(temp[6])
        asia = timezone("Asia/Kolkata")
        time.append(log.created_at.astimezone(asia).strftime("%D %H:%M:%S"))
    car_inflow.reverse()
    car_outflow.reverse()
    bike_inflow.reverse()
    bike_outflow.reverse()
    truck_inflow.reverse()
    truck_outflow.reverse()
    time.reverse()
    data = [car_inflow, car_outflow, bike_inflow, bike_outflow,
            truck_inflow, truck_outflow]
    log = VideoLog.objects.filter(moving_avg__isnull=True, video_id=video_id).order_by('-created_at')
    vehicle_count = []
    if log.exists():
        vehicle_count = log[0].data
    return HttpResponse(json.dumps({"vehicle_count": vehicle_count, "data": {"y": data, "x": time}}))


def VideoOutput(request, pk, is_move_avg):
    return render(request, 'algorithm/output_graph.html',
                  {'video_id': pk, 'is_moving_avg': True if int(is_move_avg) == 1 else False,
                   'update_time': settings.MOVING_AVERAGE_WINDOW})
