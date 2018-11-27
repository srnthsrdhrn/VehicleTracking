from pascal_voc_writer import Writer
import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa

source = "darkflow/Dataset"
output_dir = "darkflow/Data"
classes = ['small-car', 'big-car', 'bus', 'truck', 'three-wheeler', 'two-wheeler', 'lcv', 'bicycle', 'people']
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
dir_skip = 3
image_skip = 352
flag = True
for dir_count, dir in enumerate(os.listdir(source)):
    if dir_count < dir_skip:
        continue
    for file_count, file_name in enumerate(os.listdir(source + "/" + dir)):
        aug_bounding_boxes = []
        aug_classes = []
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            file_name, extension = file_name.split(".")
            if dir_count == dir_skip and int(file_name.replace("image", "")) <= image_skip:
                continue
            image = cv2.imread(source + "/" + dir + "/" + file_name + "." + extension)
            file = open(source + "/" + dir + "/" + file_name + ".txt")
            w, h, d = image.shape
            writer = Writer(output_dir + "/Images/" + dir + "_" + file_name + ".jpg", w, h)
            for line in file.readlines():
                line = line.strip()
                c, x, y, xw, yh = line.split(" ")
                c = classes[int(c)]
                x = float(x) * w
                y = float(y) * h
                xw = float(xw) * w
                yh = float(yh) * h
                xmin = round(x - xw / 2)
                xmax = round(x + xw / 2)
                ymin = round(y - yh / 2)
                ymax = round(y + yh / 2)
                if xmin > xmax:
                    print("Error at {}, {} xmin > xmax".format(dir + "_" + file_name, c))
                if ymin > ymax:
                    print("Error at {}, {} ymin > ymax".format(dir + "_" + file_name, c))
                writer.addObject(c, xmin, ymin, xmax, ymax)
                aug_classes.append(c)
                aug_bounding_boxes.append(ia.BoundingBox(x1=xmax, y1=ymax, x2=xmin, y2=ymin))
            writer.save(output_dir + "/Annotations/" + dir + "_" + file_name + ".xml")
            cv2.imwrite(output_dir + "/Images/" + dir + "_" + file_name + ".jpg", image)
            if aug_bounding_boxes.__len__() > 0:
                for id in range(0, 50):
                    try:
                        seq_det = seq.to_deterministic()
                        image1 = seq_det.augment_image(image=image)
                        bounding_boxes = seq_det.augment_bounding_boxes(
                            [ia.BoundingBoxesOnImage(aug_bounding_boxes, shape=(h, w))])[0]
                        writer = Writer("darkflow/Data/Images/{}_aug_{}_{}.jpg".format(dir, id, file_name),
                                        image1.shape[0],
                                        image1.shape[1])
                        for _class, box in zip(aug_classes, bounding_boxes.bounding_boxes):
                            writer.addObject(_class, round(box.x2), round(box.y2), round(box.x1), round(box.y1))
                        writer.save("darkflow/Data/Annotations/{}_aug_{}_{}.xml".format(dir, id, file_name))
                        cv2.imwrite("darkflow/Data/Images/{}_aug_{}_{}.jpg".format(dir, id, file_name), image1)
                    except Exception:
                        pass
            print(dir + "_" + file_name + " Parsed and Saved")
        else:
            continue
