import numpy as np
import cv2
import matplotlib.pyplot as plt


def denseStereo(leftImg, rightImg):

    h, w = leftImg.shape[:2]
    f = 0.5  # focal length
    Tx = -0.1  # translation (distance between cameras)

    # Perspective transformation matrix
    Q = np.array([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, f],
        [0, 0, -Tx, 0]
    ])

    # Brute Matching
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    depth = stereo.compute(leftImg, rightImg)
    # plt.figure()
    # plt.title('Brute Matching')
    # plt.imshow(depth)
    # plt.axis('off')

    # Semi-Global Brute Matching
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21)
    depth = stereo.compute(leftImg, rightImg)
    # plt.figure()
    # plt.title('Semi-Global Brute Matching')
    # plt.imshow(depth)
    # plt.axis('off')

    points = cv2.reprojectImageTo3D(depth, Q, handleMissingValues=False)
    points[np.isinf(points)] = 0  # remove infinities
    points = cv2.normalize(points, None, 0, 255, cv2.NORM_MINMAX)
    print(points)

    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]

    sparse_x = []
    sparse_y = []
    sparse_z = []
    for i in range(0, x.shape[0], 5):
        for j in range(0, x.shape[1], 5):
            sparse_x.append(x[i][j])
            sparse_y.append(y[i][j])
            sparse_z.append(z[i][j])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(sparse_x, sparse_y, sparse_z, cmap='gray')
    ax.view_init(45, 45)
    plt.show()


if __name__ == '__main__':
    path = '../data/'
    leftImg = cv2.imread(path + 'tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    rightImg = cv2.imread(path + 'tsukuba_r.png', cv2.IMREAD_GRAYSCALE)
    denseStereo(leftImg, rightImg)
