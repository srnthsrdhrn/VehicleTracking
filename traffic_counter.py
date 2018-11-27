import csv
import numpy as np
import os
import time
from datetime import datetime
from threading import Thread

import cv2

from Algorithm.models import Videos, VideoLog
from SmartCity import settings
from SmartCity.settings import bufferQueueDict, resultQueueDict
from custom_utils import run
from custom_utils.ArgumentsHandler import argHandler
from threading import Lock
import numpy as np

date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
# Deque(directory="media/dequeue_tmp/")
if not os.path.isdir("media/output_file"):
    os.mkdir("media/output_file")
csv_file = open("media/output_file/Video_{}.csv".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")), 'w')
csv_writer = csv.writer(csv_file)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


class DeepSenseTrafficManagement:
    def __init__(self, line_coordinates, file, video_id, path):
        """
        Various flags that control the gui working. Flagoos regarding the algorithms can be found in the run.py file
        """
        self.elapsed = 0
        self.check_counter = 0

        """
        Capturing arguments from command line
        """
        FLAGS = run.manual_setting()
        self.FLAG = argHandler()
        self.FLAG.setDefaults()
        self.FLAG.update(FLAGS)
        self.options = self.FLAG

        """
        Calling utility method input_track to initialize Sort Algorithm
        """
        self.out = None
        self.Tracker, self.encoder = self.input_track()
        self.source = self.input_source(file, path)
        self.Tracker.line_coordinate = line_coordinates
        self.moving_avg = []
        self.prev_vehicle_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.video_id = video_id
        self.wait_time = 0.05
        self.wait_count = 1
        # [start_x, start_y, end_x, end_y]

        """
        This one line initializes the Neural Network for predictions
        """
        self.exit_threads = False
        # self.init_workers()
        self.start_buffer_feeder()
        # self.start_feeder()
        self.start_collection()
        # self.start_checker(video_id)

    def input_source(self, file, path):
        if file == "IPCAM":
            if path == '0':
                camera = cv2.VideoCapture(0)
            else:
                camera = cv2.VideoCapture(path)
            date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            file = file + "_" + date
        else:
            assert os.path.isfile(path), \
                'file {} does not exist'.format(file)
            camera = cv2.VideoCapture(path)
        self.frame_rate = round(camera.get(cv2.CAP_PROP_FPS))
        if self.frame_rate == 0:
            self.frame_rate = 10
        self.frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.Tracker.frame_width = self.frame_width
        self.Tracker.frame_height = self.frame_height
        self.Tracker.frame_rate = self.frame_rate
        if self.options.testing:
            print("Test Run Acknowledged")
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            file = file.split(".")[0]
            OUTPUT_FILE_NAME = 'media\processed\{}.mp4'.format(file)
            # self.VIDEO_SCALE_RATIO = 1
            VIDEO_SCALE_RATIO = 1
            RATIO_OF_DIALOG_BOX = 0.5
            _, frame = camera.read()
            # frame = cv2.resize(frame, None, fx=self.VIDEO_SCALE_RATIO, fy=self.VIDEO_SCALE_RATIO,
            #                    interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            # frame = cv2.resize(frame, None, fx=VIDEO_SCALE_RATIO, fy=VIDEO_SCALE_RATIO, interpolation=cv2.INTER_LINEAR)
            height = frame.shape[0]
            b_width = round(frame.shape[1] * RATIO_OF_DIALOG_BOX)
            blank_image = np.zeros((height, b_width, 4), np.uint8)
            blank_image[np.where((blank_image == [0, 0, 0, 0]).all(axis=2))] = [240, 240, 240, 1]
            img = np.column_stack((frame, blank_image))
            fheight = img.shape[0]
            fwidth = img.shape[1]
            self.out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, self.frame_rate, (fwidth, fheight), True)

        blank_image = np.zeros((b_height, width, 3), np.uint8)
        blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
        img = np.row_stack((frame, blank_image))
        self.fheight = img.shape[0]
        self.fwidth = img.shape[1]

        return camera

    def raw_video(self):
        frame = None
        if self.source.isOpened:
            _, frame = self.source.read()
        return frame

    def input_track(self):
        """
        Utility function to initialize the sort algorithm
        :return: None
        """
        from sort.sort import Sort
        Tracker = Sort()
        return Tracker, None

    def start_checker(self, video_id):
        t = Thread(target=self.checker, args=(video_id,))
        t.daemon = True
        t.start()

    def checker(self, video_id):
        while True:
            video = Videos.objects.get(id=video_id)
            time.sleep(10)
            if video.processed or self.exit_threads:
                self.exit_threads = True
                # while True:
                #     try:
                #         inputQueue.get(timeout=5)
                #     except Exception:
                #         break
                while True:
                    try:
                        resultQueueDict[video_id].get(timeout=5)
                    except Exception:
                        break
                while True:
                    try:
                        bufferQueueDict[video_id].get(timeout=5)
                    except Exception:
                        break
                self.source.release()
                print("Exit Signal Received. Exiting Threads and Collecting Garbage")
                break

    def start_buffer_feeder(self):
        t = Thread(target=self.buffer_feeder)
        t.daemon = True
        t.start()

    def buffer_feeder(self):
        self.counter = 0
        # frame_flag = True

        # buffer_velocity = []
        # prev_buffer_length = 0

        while True:
            if self.exit_threads:
                print("Exiting Buffer Feeder")
                break
            # if buffer_velocity.__len__() == 5:
            #     buf_vel = sum(buffer_velocity)
            #     if buf_vel > 0 and wait_time < 1:
            #         wait_time += 0.1 * buf_vel
            #     elif sum(buffer_velocity) < 0 and wait_time > 0.1:
            #         wait_time -= 0.1 * buf_vel
            #     buffer_velocity.pop()
            # current_buffer_len = buffer_queue.qsize()
            # diff = current_buffer_len - prev_buffer_length
            # prev_buffer_length = current_buffer_len
            # if diff <= 0:
            #     buffer_velocity.insert(0, -1)
            # else:
            #     buffer_velocity.insert(0, 1)

            frame = self.raw_video()
            # if frame_flag:
            #     frame_flag = False
            #     continue
            # else:
            #     frame_flag = True
            self.counter += 1
            if frame is not None:
                bufferQueueDict[self.video_id].put([self.counter, frame])
                # print("Video: {} Buffer Q Size {}".format(self.video_id, bufferQueueDict[self.video_id].qsize()))
                videos = Videos.objects.filter(processed=True)
                if videos.exists():
                    if self.wait_count != videos.count():
                        self.wait_time = (self.wait_time * videos.count()) + 0.01
                        self.wait_count = videos.count()

            time.sleep(self.wait_time)

    def kalman_process(self):
        """
        Utility method to read from result queue in order and perform Kalman Filter Prediction
        :return: Process Image
        """

        # counter, frame = resultQueue.get()
        # return frame
        while True:
            try:
                # while True:
                counter, detections, boxes_final, imgcv = resultQueueDict[self.video_id].get(timeout=60)
                break
                # if self.check_counter + 1 == counter:
                #     break
                # else:
                #     resultQueue.put((counter, detections, boxes_final, imgcv))
                # self.check_counter = counter
            except Exception as e:
                print("Result Queue Empty and Timed out after 60 seconds " + str(e))
                return None
            # print("Result Queue size {}".format(resultQueue.qsize()))
            # print("Buffer Queue size {}".format(buffer_queue.qsize()))
        tracker = self.Tracker
        h, w, _ = imgcv.shape
        thick = int((h + w) // 300)
        line_coordinate = tracker.line_coordinate
        cv2.line(imgcv, (line_coordinate[0], line_coordinate[1]), (line_coordinate[2], line_coordinate[3]),
                 (0, 0, 255), 6)
        if detections.shape[0] == 0:
            return imgcv  # , tracker.vehicle_count[4], tracker.vehicle_count[5], tracker.vehicle_count[6]
        else:
            global csv_file, csv_writer
            timestamp = datetime.now().strftime("%H:%M:%S")
            trackers = tracker.update(detections, boxes_final, csv_file, csv_writer, timestamp, self.options.testing)
        if self.options.testing:
            for track in trackers:
                try:
                    bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
                    for box in boxes_final:
                        if (((box[2]) - 30) <= int(bbox[1]) <= ((box[2]) + 30) and ((box[0]) - 30) <= bbox[0] <= (
                                (box[0]) + 30)):
                            global v_name
                            v_name = box[4]

                    # if v_name == 'car':
                    cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  (255, 255, 255), thick // 3)
                except Exception as e:
                    print(str(e))
                # if v_name == 'bus':
                #     cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                #                   (0, 0, 255), thick // 3)
                #
                # if v_name == 'motorbike':
                #     cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                #                   (0, 255, 0), thick // 3)
        return imgcv

    def get_postprocessed(self):
        if self.Tracker.frame_count >= 1:
            self.Tracker.new_video = False
        processed_image = self.kalman_process()
        return processed_image

    def start_collection(self):
        t = Thread(target=self.update)
        t.daemon = True
        t.start()

    def update(self):
        """
        This method is the collection thread. It the processes the video from result queue and saves it
        :return: None
        """
        self.current_frame = self.get_postprocessed()
        if not os.path.isdir("media/output_Videos"):
            os.mkdir("media/output_Videos")
        if not os.path.isdir("media/output_Videos/{}/".format(self.file)):
            os.mkdir("media/output_Videos/{}".format(self.file))

        while self.current_frame is not None:
            """
            Customizations for the Saved video
            """
            if self.exit_threads:
                time.sleep(5)
                print("Exiting Source Thread")
                break
            if (self.elapsed / self.frame_rate) % 30 == 0:
                date = datetime.now().strftime("%d/%M/%Y %H:%M:%S")
                OUTPUT_FILE_NAME = "media/output_Videos/{}/{}_{}.mp4".format(self.file, self.file, date)
                if self.out:
                    self.out.release()
                self.out = cv2.VideoWriter(OUTPUT_FILE_NAME, self.fourcc, self.frame_rate, (self.fwidth, self.fheight),
                                           True)
            self.elapsed += 1
            vehicle_count = self.Tracker.vehicle_count
            if (self.elapsed / self.frame_rate) % settings.MOVING_AVERAGE_WINDOW == 0:
                VideoLog.objects.create(video_id=self.video_id, data=str(vehicle_count))
            # VehicleCount.objects.create()
            if self.options.testing:
                BASE_Y = 30
                Y_WIDTH = 50
                FONT = cv2.FONT_HERSHEY_COMPLEX
                FONT_SCALE = 0.6
                FONT_COLOR = (0, 0, 0)
                RATIO_OF_DIALOG_BOX = 0.5
                frame = self.current_frame
                # frame = cv2.resize(self.current_frame, None, fx=self.VIDEO_SCALE_RATIO, fy=self.VIDEO_SCALE_RATIO,
                #                    interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                width = frame.shape[1]
                height = frame.shape[0]
                b_width = round(frame.shape[1] * RATIO_OF_DIALOG_BOX)
                b_height = height
                blank_image = np.zeros((height, b_width, 3), np.uint8)
                blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
                overlay = np.zeros((height, width, 4), dtype='uint8')
                img_path = 'icons/bosch.png'
                logo = cv2.imread(img_path, -1)
                watermark = image_resize(logo, height=50)
                watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
                watermark_h, watermark_w, watermark_c = watermark.shape
                for i in range(0, watermark_h):
                    for j in range(0, watermark_w):
                        if watermark[i, j][3] != 0:
                            offset = 10
                            h_offset = height - watermark_h - offset
                            w_offset = height - watermark_w - offset
                            overlay[10 + i, 10 + j] = watermark[i, j]
                cv2.putText(frame, "DeepSense", (width - int(width * 0.25), round(height * 0.1)), FONT, 1,
                            (255, 255, 255), 2)
                cv2.addWeighted(overlay, 1, frame, 1, 0, frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                GRID_WIDTH = 200
                PADDING = 10
                INITIAL = PADDING + GRID_WIDTH + 50
                cv2.rectangle(blank_image, (PADDING, PADDING), (b_width - PADDING, b_height - PADDING), (0, 0, 0), 1)
                cv2.line(blank_image, (INITIAL, 10), (INITIAL, b_height - PADDING), (0, 0, 0), 1)
                cv2.line(blank_image, (INITIAL + GRID_WIDTH * 1, 10), (INITIAL + GRID_WIDTH * 1, b_height - PADDING),
                         (0, 0, 0), 1)
                cv2.line(blank_image, (INITIAL + GRID_WIDTH * 2, 10), (INITIAL + GRID_WIDTH * 2, b_height - PADDING),
                         (0, 0, 0), 1)
                # # Initial Data
                cv2.putText(blank_image, 'Vehicle Type', (50, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
                cv2.putText(blank_image, 'In Flow', (330, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
                cv2.putText(blank_image, 'Out Flow', (530, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
                cv2.putText(blank_image, 'Total', (750, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)

                for id, obj in enumerate(self.options.trackObj):
                    cv2.putText(blank_image, obj, (50, BASE_Y + Y_WIDTH * (1 + id)), FONT, FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][1]), (330, BASE_Y + Y_WIDTH * 1 + id), FONT,
                                FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][0]), (530, BASE_Y + Y_WIDTH * 1 + id), FONT,
                                FONT_SCALE, FONT_COLOR, 1)
                    cv2.putText(blank_image, str(vehicle_count[obj][0] + vehicle_count[obj][1]),
                                (750, BASE_Y + Y_WIDTH * 1), FONT, FONT_SCALE, FONT_COLOR, 1 + id)

                # cv2.putText(blank_image, 'big-car', (50, BASE_Y + Y_WIDTH * 2), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, str(vehicle_count['big-car'][1]), (330, BASE_Y + Y_WIDTH * 2), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, str(vehicle_count['big-car'][0]), (530, BASE_Y + Y_WIDTH * 2), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, str(vehicle_count['big-car'][0] + vehicle_count['big-car'][1]), (750, BASE_Y + Y_WIDTH * 2), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'bus', (50, BASE_Y + Y_WIDTH * 3), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '', (330, BASE_Y + Y_WIDTH * 3), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 3), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, str(vehicle_count['bus'][0] + vehicle_count['bus'][1]), (750, BASE_Y + Y_WIDTH * 3), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'truck', (50, BASE_Y + Y_WIDTH * 4), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 4), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 4), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 4), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'three-wheeler', (50, BASE_Y + Y_WIDTH * 5), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 5), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 5), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 5), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'two-wheeler', (50, BASE_Y + Y_WIDTH * 6), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 6), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 6), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 6), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'lcv', (50, BASE_Y + Y_WIDTH * 7), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 7), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 7), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 7), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'bicycle', (50, BASE_Y + Y_WIDTH * 8), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 8), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 8), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 8), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'people', (50, BASE_Y + Y_WIDTH * 9), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 9), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 9), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 9), FONT, FONT_SCALE, FONT_COLOR, 1)
                #
                # cv2.putText(blank_image, 'auto-rickshaw', (50, BASE_Y + Y_WIDTH * 10), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '10', (330, BASE_Y + Y_WIDTH * 10), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '20', (530, BASE_Y + Y_WIDTH * 10), FONT, FONT_SCALE, FONT_COLOR, 1)
                # cv2.putText(blank_image, '30', (750, BASE_Y + Y_WIDTH * 10), FONT, FONT_SCALE, FONT_COLOR, 1)

                img = np.column_stack((frame, blank_image))
                self.out.write(img)
            self.current_frame = self.get_postprocessed()
        self.exit_threads = True
        self.out.release()
        csv_file.close()
        self.source.release()
        if not self.exit_threads:
            print("Finished Processing Video Exiting Source Thread")
