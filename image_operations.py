import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def image_blend(source, image, x, y):
    (h, w) = image.shape[:2]
    source[y:y + h, x:x + w, :] = image[0:h, 0:w, :]
    return source


def change_brightness(img, modifier):
    return cv2.add(img, np.array([float(modifier)]))


def add_glow(img, glow_strength=1, glow_radius=25):
    img_blurred = cv2.GaussianBlur(img, (glow_radius, glow_radius), 1)
    img_blended = cv2.addWeighted(img, 1, img_blurred, glow_strength, 0)
    return img_blended

