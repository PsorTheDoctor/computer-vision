import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

path = '../data/'


def savePointCloud(filename, points, colors, max_z):
    pcd = o3d.geometry.PointCloud()
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            xyz = points[i, j]
            bgr = colors[i, j]

            # If points are too far away don't take them into account
            if abs(xyz[2]) < max_z:
                pcd.points.append([xyz[0], xyz[1], xyz[2]])
                pcd.colors.append([bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0])

    o3d.io.write_point_cloud(filename, pcd)


def visualizePointCloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2
    vis.run()
    vis.destroy_window()


def denseStereo(leftImg, rightImg):
    h, w = leftImg.shape[:2]
    f = 500  # focal length
    Tx = -100  # translation (distance) between cameras
    Q = np.array([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, f],
        [0, 0, -1/Tx, 0]
    ])
    colors = leftImg
    leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # Brute Matching
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    depth = stereo.compute(leftImg, rightImg)
    depthNorm = cv2.normalize(depth, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.title('Brute Matching')
    plt.imshow(depthNorm)
    plt.axis('off')

    # Semi-Global Brute Matching
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21)
    depth = stereo.compute(leftImg, rightImg)
    depthNorm = cv2.normalize(depth, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.title('Semi-Global Brute Matching')
    plt.imshow(depthNorm)
    plt.axis('off')
    plt.show()

    points = cv2.reprojectImageTo3D(depth, Q, handleMissingValues=False)
    savePointCloud('cloud.pcd', points, colors, max_z=500)
    visualizePointCloud('cloud.pcd')


if __name__ == '__main__':
    leftImg = cv2.imread(path + 'tsukuba_l.png')
    rightImg = cv2.imread(path + 'tsukuba_r.png')
    denseStereo(leftImg, rightImg)
