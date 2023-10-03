import cv2
import numpy as np

corners = []
w_A4, h_A4 = 210, 297


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


def compute_homography(sheet_corners):
    warped_sheet_corners = np.array([
        [0, 0], [0, h_A4], [w_A4, h_A4], [w_A4, 0]
    ], dtype=np.float32)
    warped_sheet_corners *= 2.0

    H, _ = cv2.findHomography(
        sheet_corners, warped_sheet_corners, cv2.RANSAC, 5.0
    )
    # H = cv2.getPerspectiveTransform(sheet_corners, sheet_dst_corners)
    return H


def compute_pcb_width_height(pcb_corners, H):
    warped_pcb_corners = []

    for i in range(len(pcb_corners)):
        pcb_corner = np.array([pcb_corners[i, 0], pcb_corners[i, 1], 1])
        p0 = H @ pcb_corner
        warped_pcb_corners.append([p0[0] / p0[2], p0[1] / p0[2]])

    warped_pcb_corners = np.array(warped_pcb_corners)
    w = cv2.norm(warped_pcb_corners[0] - warped_pcb_corners[1], cv2.NORM_L2) / 2.0
    h = cv2.norm(warped_pcb_corners[0] - warped_pcb_corners[2], cv2.NORM_L2) / 2.0
    return w, h


if __name__ == '__main__':
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

    H = compute_homography(sheet_corners)
    print('Homography: \n', H)

    warped_img = cv2.warpPerspective(img, H, (w_A4*2, h_A4*2))

    w, h = compute_pcb_width_height(pcb_corners, H)
    print(f'Width: {int(w)}mm height: {int(h)}mm')

    cv2.imshow(window, warped_img)
    cv2.waitKey(0)
