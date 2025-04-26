import cv2
import numpy as np

img = cv2.imread('../data/adam.jpg')

h, w = img.shape[:2]
focal_length = w

cam_matrix = np.array([
    [focal_length, 0, w // 2],
    [0, focal_length, h // 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float32)

new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(
    cam_matrix, dist_coeffs, (w, h), 1, (w, h)
)

undistorted = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_cam_matrix)

cv2.imshow('', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
