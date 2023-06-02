import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

intristicMat = np.array([
  [514.682, 0, 320, 0],
  [0, 514.682, 240, 0],
  [0, 0, 1, 0]], dtype=np.float32)

extrinsicMat = np.array([
  [-1, 0, 0, -0.1],
  [0, 0.9063, -0.4226, -0.1994],
  [0, -0.4226, -0.9063, 1.3067],
  [0, 0, 0, 1]], dtype=np.float32)


def findObjectByColor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerMask = cv2.inRange(hsv, (0, 100, 20), (10, 255, 255))
    upperMask = cv2.inRange(hsv, (160, 100, 20), (179, 255, 255))
    mask = lowerMask + upperMask
    mask = cv2.erode(mask, (3, 3))
    mask = cv2.dilate(mask, (3, 3))
    return mask


def findCenter(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    point3d = np.zeros(3)
    if len(cnts) > 0:
        area = max(cnts, key=cv2.contourArea)
        center, radius = cv2.minEnclosingCircle(area)
        point3d = np.array([center[0], center[1], radius], dtype=np.float32)
    return point3d


def computeOpticalCenter(Pp):
    C = -np.linalg.inv(Pp[0:3, 0:3]) @ Pp[0:3, 3:4]
    return np.vstack((C, 1))


def computeFundamentalMat(epipole, projMat):
    erx = np.zeros((3, 3))
    erx[0, 1] = -epipole[2]
    erx[0, 2] = epipole[1]
    erx[1, 0] = epipole[2]
    erx[1, 2] = epipole[0]
    erx[2, 0] = epipole[1]
    erx[2, 1] = epipole[0]
    return erx @ projMat @ np.linalg.inv(np.vstack((projMat, [0, 0, 0, 1])))


def computePluckerLine(M1, M2):
    plucker = np.zeros(2)
    plucker[0] = np.cross(M1, M2) / np.linalg.norm(M2)
    plucker[1] = M2 / np.linalg.norm(M2)
    return plucker


def computePluckerIntersect(plucker1, plucker2):
    mu1 = plucker1[0]
    mu2 = plucker2[0]
    v1 = plucker1[1]
    v2 = plucker2[1]

    v1_v2xmu2 = v1 @ np.cross(v2, mu2)
    v1v2_v1_v2xmu1 = v1 @ v2 * (v1 @ np.cross(v2, mu1))
    pow_v1xv2 = np.pow(np.linalg.norm(np.cross(v1, v2)), 2)
    M1 = (v1_v2xmu2 - v1v2_v1_v2xmu1) / pow_v1xv2 * v1 + np.cross(v1, mu1)

    v2_v1xmu1 = v2 @ np.cross(v1, mu1)
    v2v1_v2_v1xmu2 = v2 @ v2 * (v2 @ np.cross(v2, mu2))
    pow_v2xv1 = np.pow(np.linalg.norm(np.cross(v2, v1)), 2)
    M2 = (v2_v1xmu1 - v2v1_v2_v1xmu2) / pow_v2xv1 * v2 + np.cross(v2, mu2)
    return M1 + (M2 - M1) / 2


if __name__ == '__main__':
    left = cv2.imread('left.png')
    right = cv2.imread('right.png')

    leftMask = findObjectByColor(left)
    rightMask = findObjectByColor(right)
    # plt.figure()
    # plt.imshow(leftMask)
    # plt.figure()
    # plt.imshow(rightMask)
    # plt.show()

    leftCenter = findCenter(leftMask)
    rightCenter = findCenter(rightMask)

    leftPoint = [leftCenter[0], leftCenter[1], 1]
    rightPoint = [rightCenter[0], rightCenter[1], 1]

    H = extrinsicMat
    KA = intristicMat
    proj = KA @ H
    print('Projection matrix')
    print(proj)

    # Pp = np.hstack((proj[0:3, 0:3], proj[0:3, 3:4]))
    # print(Pp)

    C = computeOpticalCenter(proj)
    epipole = proj @ C
    print('\nEpipole')
    print(epipole)

    F = computeFundamentalMat(epipole, proj)
    print('\nFundamental matrix')
    print(F)

    epipolarLine = F @ np.hstack((leftCenter, 1)).T
    print('\nEpipolar line')
    print(epipolarLine)
