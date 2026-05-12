import cv2
from skimage.filters import meijering, sato, frangi, hessian

img = cv2.imread('../data/strawberries.jpg')
img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
vesselness = frangi(img, sigmas=(5, 6, 7, 8, 9, 10))

cv2.imshow('', vesselness)
cv2.waitKey(0)
cv2.destroyAllWindows()
