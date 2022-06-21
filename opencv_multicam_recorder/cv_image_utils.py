import os
import sys
import cv2
import numpy as np


def cat_horiz(image1, image2, img_sep=5, scale=False):
    if scale:
        high = np.maximum(image1.shape[0], image2.shape[0])
        low = np.minimum(image1.shape[0], image2.shape[0])
        scale_fac = float(high) / float(low)
        if image1.shape[0] < image2.shape[0]:
            im_size = (int(image1.shape[1] * scale_fac), int(image1.shape[0] * scale_fac))
            image1 = cv2.resize(image1, dsize=im_size)
        elif image1.shape[0] > image2.shape[0]:
            im_size = (int(image2.shape[1] * scale_fac), int(image2.shape[0] * scale_fac))
            image2 = cv2.resize(image2, dsize=im_size)

    cat_shape = [np.maximum(image1.shape[0], image2.shape[0]),
                 image1.shape[1] + image2.shape[1] + img_sep,
                 image2.shape[2]]

    frames = np.zeros(cat_shape, dtype=image1.dtype)
    frames[0:image1.shape[0], 0:image1.shape[1]] = image1
    frames[0:image2.shape[0], image1.shape[1] + img_sep:] = image2
    return frames


def cat_vert(image1, image2, img_sep=5, scale=False):
    if scale:
        high = np.maximum(image1.shape[1], image2.shape[1])
        low = np.minimum(image1.shape[1], image2.shape[1])
        scale_fac = float(high) / float(low)
        if image1.shape[0] < image2.shape[0]:
            im_size = (int(image1.shape[1] * scale_fac), int(image1.shape[0] * scale_fac))
            image1 = cv2.resize(image1, dsize=im_size)
        elif image1.shape[0] > image2.shape[0]:
            im_size = (int(image2.shape[1] * scale_fac), int(image2.shape[0] * scale_fac))
            image2 = cv2.resize(image2, dsize=im_size)

    cat_shape = [image1.shape[0] + image2.shape[0] + img_sep,
                 np.maximum(image1.shape[1], image2.shape[1]),
                 image2.shape[2]]

    frames = np.zeros(cat_shape, dtype=image1.dtype)
    frames[0:image1.shape[0], 0:image1.shape[1]] = image1
    frames[image1.shape[0] + img_sep:, 0:image2.shape[1]] = image2
    return frames


def put_text_multiline(img, text, pos=(20, 20), font=cv2.FONT_HERSHEY_SIMPLEX,
                       scale=0.5, color=(255, 128, 255), thickness=2, line_advance=20):
    posl = [pos[0], pos[1]]
    for l in text.split("\n"):
        img = cv2.putText(img, l, (posl[0], posl[1]), font, scale, color, thickness, lineType=cv2.LINE_AA)
        posl[1] += line_advance
    return img


def get_camera_name(device_id: str):
    return os.popen(f"cat /sys/class/video4linux/"
                    f"{device_id.split(os.sep)[-1]}/name").read().strip()
