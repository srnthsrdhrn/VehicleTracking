import csv
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

date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
# Deque(directory="media/dequeue_tmp/")
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
        # [start_x, start_y, end_x, end_y]

        """
        This one line initializes the Neural Network for predictions
        """
        self.exit_threads = False
        # self.init_workers()
        self.start_buffer_feeder()
        # self.start_feeder()
        self.start_collection()
        self.start_checker(video_id)
        self.video_id = video_id

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
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # # date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        # file = file.split(".")[0]
        # OUTPUT_FILE_NAME = 'media\processed\{}.mp4'.format(file)
        # # self.VIDEO_SCALE_RATIO = 1
        # RATIO_OF_BELOW_BOX = 0.35
        # _, frame = camera.read()
        # # frame = cv2.resize(frame, None, fx=self.VIDEO_SCALE_RATIO, fy=self.VIDEO_SCALE_RATIO,
        # #                    interpolation=cv2.INTER_LINEAR)
        # width = frame.shape[1]
        # height = frame.shape[0]
        # b_height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)
        #
        # blank_image = np.zeros((b_height, width, 3), np.uint8)
        # blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
        # img = np.row_stack((frame, blank_image))
        # fheight = img.shape[0]
        # fwidth = img.shape[1]
        # self.out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, self.frame_rate, (fwidth, fheight), True)

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
        wait_time = 0.05
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
                bufferQueueDict[self.video_id].put((self.counter, frame))
            time.sleep(wait_time)

    def kalman_process(self):
        """
        Utility method to read from result queue in order and perform Kalman Filter Prediction
        :return: Process Image
        """

        # counter, frame = resultQueue.get()
        # return frame
        try:
            # while True:
            counter, detections, boxes_final, imgcv = resultQueueDict[self.video_id].get()
            # if self.check_counter + 1 == counter:
            #     break
            # else:
            #     resultQueue.put((counter, detections, boxes_final, imgcv))
            # self.check_counter = counter
        except Exception as e:
            print("Result Queue Empty and Timed out after 30 seconds " + str(e))
            return None
        # print("Result Queue size {}".format(resultQueue.qsize()))
        # print("Buffer Queue size {}".format(buffer_queue.qsize()))
        tracker = self.Tracker
        h, w, _ = imgcv.shape
        # thick = int((h + w) // 300)
        line_coordinate = tracker.line_coordinate
        cv2.line(imgcv, (line_coordinate[0], line_coordinate[1]), (line_coordinate[2], line_coordinate[3]),
                 (0, 0, 255), 6)
        if detections.shape[0] == 0:
            return imgcv  # , tracker.vehicle_count[4], tracker.vehicle_count[5], tracker.vehicle_count[6]
        else:
            trackers = tracker.update(detections, boxes_final)
        # for track in trackers:
        #     bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
        #     for i in range(len(boxes_final)):
        #         box = boxes_final[i]
        #
        #         if (((box[2]) - 30) <= int(bbox[1]) <= ((box[2]) + 30) and ((box[0]) - 30) <= bbox[0] <= (
        #                 (box[0]) + 30)):
        #             global v_name
        #             v_name = box[4]
        #
        #     if v_name == 'car':
        #         cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
        #                       (255, 255, 255), thick // 3)
        #     if v_name == 'bus':
        #         cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
        #                       (0, 0, 255), thick // 3)
        #
        #     if v_name == 'motorbike':
        #         cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
        #                       (0, 255, 0), thick // 3)
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
        global csv_file, csv_writer
        self.current_frame = self.get_postprocessed()
        while self.current_frame is not None:
            """
            Customizations for the Saved video
            """
            if self.exit_threads:
                time.sleep(5)
                print("Exiting Source Thread")
                break
            self.elapsed += 1
            vehicle_count = self.Tracker.vehicle_count
            if (self.elapsed / self.frame_rate) % settings.MOVING_AVERAGE_WINDOW == 0:
                VideoLog.objects.create(video_id=self.video_id, moving_avg=str(vehicle_count[7:]))
            if (self.elapsed / self.frame_rate) % settings.MOVING_AVERAGE_WINDOW == 0:
                csv_writer.writerow(vehicle_count)
                csv_file.flush()
                VideoLog.objects.create(video_id=self.video_id, data=str(vehicle_count))
            # VehicleCount.objects.create()
            # FONT = cv2.FONT_HERSHEY_SIMPLEX
            # FONT_SCALE = 0.4
            # FONT_SCALE_HEADING = 0.6
            # FONT_COLOR = (0, 0, 0)
            # RATIO_OF_BELOW_BOX = 0.35
            # frame = self.current_frame
            # # frame = cv2.resize(self.current_frame, None, fx=self.VIDEO_SCALE_RATIO, fy=self.VIDEO_SCALE_RATIO,
            # #                    interpolation=cv2.INTER_LINEAR)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            # width = frame.shape[1]
            # height = frame.shape[0]
            # img_path = 'icons/bosch.png'
            # logo = cv2.imread(img_path, -1)
            # watermark = image_resize(logo, height=50)
            # watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
            # overlay = np.zeros((height, width, 4), dtype='uint8')
            # watermark_h, watermark_w, watermark_c = watermark.shape
            # for i in range(0, watermark_h):
            #     for j in range(0, watermark_w):
            #         if watermark[i, j][3] != 0:
            #             overlay[10 + i, 10 + j] = watermark[i, j]
            # cv2.addWeighted(overlay, 1, frame, 1, 0, frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # width = frame.shape[1]
            # height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)
            # blank_image = np.zeros((height, width, 3), np.uint8)
            # blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
            # cv2.putText(frame, "DeepSense", (width - int(width * 0.25), round(height * 0.2)), FONT, 1,
            #             (255, 255, 255), 2)
            # """
            # This part of code displays the algorithm's output to the canvas
            # """
            #
            # self.elapsed += 1
            #
            # vehicle_count = self.Tracker.vehicle_count
            # """
            # Adding text to Output Video
            # """
            # # Car Data
            # cv2.putText(blank_image, 'Vehicle Type', (30, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Count (Y/N)', (30, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Toll(Y/N)', (30, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'In Flow', (30, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Out Flow', (30, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Avg In Flow', (30, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Avg Out Flow', (30, 150), FONT, FONT_SCALE, FONT_COLOR, 1)
            #
            # # Car Data:
            # cv2.putText(blank_image, 'Car', (180, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Yes', (180, 50), FONT, FONT_SCALE, FONT_COLOR,
            #             1)
            # cv2.putText(blank_image, 'No', (180, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[0]), (180, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[4]), (180, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[7]), (180, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[10]), (180, 150), FONT, FONT_SCALE, FONT_COLOR,
            #             1)
            #
            # # Bus Data:
            # cv2.putText(blank_image, 'Bus', (255, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Yes', (255, 50), FONT, FONT_SCALE, FONT_COLOR,
            #             1)
            # cv2.putText(blank_image, 'No', (255, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[1]), (255, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[5]), (255, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[8]), (255, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[11]), (255, 150), FONT, FONT_SCALE, FONT_COLOR,
            #             1)
            #
            # # Bike Data:
            # cv2.putText(blank_image, 'Bike', (330, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'Yes', (330, 50), FONT, FONT_SCALE,
            #             FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (330, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[2]), (330, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[6]), (330, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[9]), (330, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '{}'.format(vehicle_count[12]), (330, 150), FONT, FONT_SCALE, FONT_COLOR,
            #             1)
            #
            # # Truck Data:
            # cv2.putText(blank_image, 'Truck', (405, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (405, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (405, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (405, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (405, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (405, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (405, 150), FONT, FONT_SCALE, FONT_COLOR, 1)
            #
            # # Rickshaw Data:
            # cv2.putText(blank_image, 'Three Wheeler', (480, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (480, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (480, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (480, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (480, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (480, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (480, 150), FONT, FONT_SCALE, FONT_COLOR, 1)
            #
            # # Tractor Data:
            # cv2.putText(blank_image, 'Tractor', (630, 30), FONT, FONT_SCALE_HEADING, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (630, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, 'No', (630, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (630, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (630, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (630, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
            # cv2.putText(blank_image, '0', (630, 150), FONT, FONT_SCALE, FONT_COLOR, 1)
            # img = np.row_stack((frame, blank_image))
            # self.out.write(img)
            self.current_frame = self.get_postprocessed()
        # self.exit_threads = True
        # self.out.release()
        csv_file.close()
        self.source.release()
        if not self.exit_threads:
            print("Finished Processing Video Exiting Source Thread")
