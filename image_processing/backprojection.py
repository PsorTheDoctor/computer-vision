import cv2
import numpy as np


def backproj(bins):
    histSize = max(bins, 2)
    hueRange = [0, 180]
    hist = cv2.calcHist([hue], [0], None, [histSize], hueRange, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=255,  norm_type=cv2.NORM_MINMAX)
    backproj = cv2.calcBackProject([hue], [0], hist, hueRange, scale=1)
    cv2.imshow('Back Projection', backproj)


img = cv2.imread('../data/sunflower.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ch = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv2.mixChannels([hsv], [hue], ch)

window = 'Source image'
cv2.namedWindow(window)
bins = 25
cv2.createTrackbar('Hue bins: ', window, bins, 180, backproj)
backproj(bins)
cv2.imshow(window, img)
cv2.waitKey()
