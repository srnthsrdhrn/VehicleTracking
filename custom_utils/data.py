import numpy as np
import cv2
from datetime import datetime


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
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


FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (0, 0, 0)
VIDEO_SCALE_RATIO = 1
RATIO_OF_DIALOG_BOX = 0.5
date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
OUTPUT_FILE_NAME = 'C:\\Users\\fud1cob\Desktop\Srinath\Work\SmartCity\media\Video\\test.mp4'
cap = cv2.VideoCapture("C:\\Users\\fud1cob\Desktop\Srinath\Work\SmartCity\media\\IISC1.mp4")
frame = 1
img_path = 'C:\\Users\\fud1cob\Desktop\Srinath\Work\SmartCity\icons\\bosch.png'
logo = cv2.imread(img_path, -1)
watermark = image_resize(logo, height=50)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
out = None
BASE_Y = 30
Y_WIDTH = 50
while frame is not None:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame = cv2.resize(frame, None, fx=VIDEO_SCALE_RATIO, fy=VIDEO_SCALE_RATIO, interpolation=cv2.INTER_LINEAR)
    width = frame.shape[1]
    height = frame.shape[0]
    b_width = round(frame.shape[1] * RATIO_OF_DIALOG_BOX)
    b_height = height
    # b_height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)
    blank_image = np.zeros((height, b_width, 3), np.uint8)
    blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
    overlay = np.zeros((height, width, 4), dtype='uint8')
    watermark_h, watermark_w, watermark_c = watermark.shape
    for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if watermark[i, j][3] != 0:
                offset = 10
                h_offset = height - watermark_h - offset
                w_offset = height - watermark_w - offset
                overlay[10 + i, 10 + j] = watermark[i, j]
    cv2.putText(frame, "DeepSense", (width - int(width * 0.25), round(height * 0.1)), FONT, 1, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 1, frame, 1, 0, frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    GRID_WIDTH = 200
    PADDING = 10
    INITIAL = PADDING + GRID_WIDTH + 50
    cv2.rectangle(blank_image, (PADDING, PADDING), (b_width - PADDING, b_height - PADDING), (0, 0, 0), 1)
    cv2.line(blank_image, (INITIAL, 10), (INITIAL, b_height - PADDING), (0, 0, 0), 1)
    cv2.line(blank_image, (INITIAL + GRID_WIDTH * 1, 10), (INITIAL + GRID_WIDTH * 1, b_height - PADDING), (0, 0, 0), 1)
    cv2.line(blank_image, (INITIAL + GRID_WIDTH * 2, 10), (INITIAL + GRID_WIDTH * 2, b_height - PADDING), (0, 0, 0), 1)
    # small - car
    # big - car
    # bus
    # truck
    # three - wheeler
    # two - wheeler
    # lcv
    # bicycle
    # people
    # auto - rickshaw

    # # Initial Data
    cv2.putText(blank_image, 'Vehicle Type', (50, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'In Flow', (330, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Out Flow', (530, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Total', (750, BASE_Y), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'small-car', (50, BASE_Y+Y_WIDTH*1), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*1), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*1), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*1), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'big-car', (50, BASE_Y+Y_WIDTH*2), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*2), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*2), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*2), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'bus', (50, BASE_Y+Y_WIDTH*3), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*3), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*3), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*3), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'truck', (50, BASE_Y+Y_WIDTH*4), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*4), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*4), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*4), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'three-wheeler', (50, BASE_Y+Y_WIDTH*5), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*5), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*5), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*5), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'two-wheeler', (50, BASE_Y+Y_WIDTH*6), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*6), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*6), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*6), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'lcv', (50, BASE_Y+Y_WIDTH*7), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*7), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*7), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*7), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'bicycle', (50, BASE_Y+Y_WIDTH*8), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*8), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*8), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*8), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'people', (50, BASE_Y+Y_WIDTH*9), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*9), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*9), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*9), FONT, FONT_SCALE, FONT_COLOR, 1)

    cv2.putText(blank_image, 'auto-rickshaw', (50, BASE_Y+Y_WIDTH*10), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '10', (330, BASE_Y+Y_WIDTH*10), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '20', (530, BASE_Y+Y_WIDTH*10), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '30', (750, BASE_Y+Y_WIDTH*10), FONT, FONT_SCALE, FONT_COLOR, 1)

    img = np.column_stack((frame, blank_image))
    fheight = img.shape[0]
    fwidth = img.shape[1]
    cv2.imwrite("test.jpg", img)
    break
    # if out is None:
    #     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, fps, (fwidth, fheight), True)
    # else:
    #     out.write(img)
    # cv2.imshow("Image", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# out.release()
cap.release()
