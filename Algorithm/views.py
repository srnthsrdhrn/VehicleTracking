import time
from threading import Thread

import cv2
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect

# Create your views here.
from django.urls import reverse
from django.views.generic import ListView, UpdateView, DeleteView

from Algorithm.forms import VideoProcessForm
from Algorithm.models import Videos


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
    line_coordinates = [video.line_coord_init_x, video.line_coord_init_y, video.line_coord_end_x,
                        video.line_coord_end_y]
    path = video.file.path if video.file else video.ip_link
    file = video.file.name if video.file else "IPCAM"
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


# class DepartmentUpdate(UpdateView):
#     model = Videos
#     fields = ['username', 'first_name', 'last_name', 'email', 'institution', 'academic', 'code', 'rfidvalue']
#     template_name = 'algorithm/update.html'
#
#     def get_success_url(self):
#         return reverse('list_department')


class DeleteVideo(DeleteView):
    model = Videos
    template_name = 'algorithm/delete.html'

    def get_success_url(self):
        return reverse('list_video')
