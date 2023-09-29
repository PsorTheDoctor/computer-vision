import cv2
import numpy as np

corners = []


def mouse_callback(event, x, y, flags, param):
    global corners
    # If left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x, y))


def select_corners(window, img):
    img_copy = img.copy()
    n_corners = 8

    while len(corners) < n_corners:
        # Listening to the mouse events
        cv2.setMouseCallback(window, mouse_callback)

        # Adding a new point to the image
        if len(corners) > 0:
            cv2.circle(img_copy, corners[len(corners) - 1],
                radius=5, color=(0, 0, 255), thickness=-1
            )
        cv2.imshow(window, img_copy)
        if cv2.waitKey(50) == 27:
            break


img = cv2.imread('../data/PCB.jpg')
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
img = cv2.putText(img,
    'Select 4 paper sheet corners and 4 PCB board corners',
    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
)
window = 'Measuring with homography'
cv2.namedWindow(window)

select_corners(window, img)

sheet_corners = np.array(corners[:4], dtype=np.float32)
pcb_corners = np.array(corners[4:], dtype=np.float32)

w_A4, h_A4 = 210, 297
flatten_sheet_corners = np.array([
    [0, 0], [0, h_A4], [w_A4, h_A4], [w_A4, 0]
], dtype=np.float32)
flatten_sheet_corners *= 2.0

H, _ = cv2.findHomography(
    sheet_corners, flatten_sheet_corners, cv2.RANSAC, 5.0
)
# H = cv2.getPerspectiveTransform(sheet_corners, sheet_dst_corners)
print(H)

warped_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

pcb_corners = np.hstack((pcb_corners, np.ones((4, 1))))
flatten_pcb_corners = H @ pcb_corners.T

for i in range(4):
    print(f'{int(flatten_pcb_corners[1][i])} {int(flatten_pcb_corners[0][i])}')
    # cv2.circle(warped_img,
    #     (int(flatten_pcb_corners[0][i]), int(flatten_pcb_corners[1][i])),
    #     radius=5, color=(0, 0, 255), thickness=-1
    # )

cv2.imshow(window, warped_img)
cv2.waitKey(0)
