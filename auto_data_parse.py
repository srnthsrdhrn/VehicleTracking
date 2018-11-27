from pascal_voc_writer import Writer
import cv2
import json
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

bbs = json.loads(open('darkflow/Auto/Auto/bbs/bbs.json').read())

_class = 'auto-rickshaw'
count = 0
import os

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
], random_order=True)
limit = 95
for i in range(0, len(bbs)):
    try:
        if i <= limit:
            continue
        image_path = "darkflow/Auto/Auto/images/" + str(i) + '.jpg'
        image = cv2.imread(image_path)
        w, h, d = image.shape
        writer = Writer("darkflow/Data/Images/auto_{}.jpg".format(i), w, h)
        boxes = bbs[i]
        f = False
        t = image
        aug_bounding_boxes = []
        for j in range(len(boxes)):
            flag = False
            x, y = boxes[j][0]
            h = boxes[j][-1][1] - y
            w = boxes[j][1][0] - x
            # print x,y, h, w
            # print boxes[j]
            xmin = round(x)
            xmax = round(x + w)
            ymin = round(y)
            ymax = round(y + h)
            if xmin > xmax:
                # print("Error at {}, {} xmin > xmax".format(i, j))
                flag = True
                f = True
            if ymin > ymax:
                f = True
                flag = True
                # print("Error at {}, {} ymin > ymax".format(i, j))
            if not flag:
                writer.addObject(_class, round(xmin), round(ymin), round(xmax), round(ymax))
                aug_bounding_boxes.append(ia.BoundingBox(x1=xmax, y1=ymax, x2=xmin, y2=ymin))
        if not f:
            writer.save("darkflow/Data/Annotations/auto_{}.xml".format(i))
            cv2.imwrite("darkflow/Data/Images/auto_{}.jpg".format(i), image)
        if aug_bounding_boxes.__len__() > 0:
            for id in range(0, 50):
                seq_det = seq.to_deterministic()
                image1 = seq_det.augment_image(image=image)
                bounding_boxes = seq_det.augment_bounding_boxes(
                    [ia.BoundingBoxesOnImage(aug_bounding_boxes, shape=(h, w))])[0]
                writer = Writer("darkflow/Data/Images/auto_{}_aug_{}.jpg".format(i, id), image1.shape[0], image1.shape[1])
                for box in bounding_boxes.bounding_boxes:
                    writer.addObject(_class, round(box.x2), round(box.y2), round(box.x1), round(box.y1))
                if not f:
                    writer.save("darkflow/Data/Annotations/auto_{}_aug_{}.xml".format(i, id))
                    cv2.imwrite("darkflow/Data/Images/auto_{}_aug_{}.jpg".format(i, id), image1)
    except Exception as e:
        print(str(e))
        continue
    # else:
    # os.remove("darkflow/Data/Images/auto_{}.jpg".format(i))
    # cv2.imwrite("darkflow/Data/Images/auto_{}.jpg".format(i), image)
    print(str(i) + " Processed")
# print("count: {}".format(count))
