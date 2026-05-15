import cv2
import numpy as np
import open3d as o3d


def load_video(path, max_frames=100, step=5):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def get_intrinsics(img):
    h, w = img.shape[:2]
    f = 1000  # focal length
    cx, cy = w // 2, h // 2
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])


def detect_and_match(img1, img2, draw_matches=True):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    FLANN_INDEX_LSH = 6
    idx_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if draw_matches:
        output_img = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(img1, kp1, img2, kp2, good_matches, output_img)
        cv2.imshow('', output_img)
        cv2.waitKey(50)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2


def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, pts1, pts2


def rectify(img1, img2, K, R, t):
    h, w = img1.shape[:2]
    dist = np.zeros(5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, (w, h), R, t)

    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    return rect1, rect2, Q


def compute_disparity(rect1, rect2):
    block_size = 9
    P1 = 8 * block_size ** 2
    P2 = 32 * block_size ** 2
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=block_size, P1=P1, P2=P2
    )
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    return stereo.compute(gray1, gray2).astype(np.float32) / 16.0


def disparity_to_point_cloud(disp, img, Q):
    pts = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = (disp > 1.0) & np.isfinite(pts).all(axis=2)
    return pts[mask], colors[mask]


def point_cloud(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.visualization.draw_geometries([pcd], window_name='Dense SfM')


def run_dense_sfm(path):
    frames = load_video(path)
    K = get_intrinsics(frames[0])
    global_pts = []
    global_colors = []
    for i in range(len(frames) - 1):
        img1, img2 = frames[i], frames [i + 1]
        pts1, pts2 = detect_and_match(img1, img2)
        R, t, pts1, pts2 = estimate_pose(pts1, pts2, K)
        rect1, rect2, Q = rectify(img1, img2, K, R, t)
        disp = compute_disparity(rect1, rect2)
        pts, colors = disparity_to_point_cloud(disp, rect1, Q)
        global_pts.append(pts)
        global_colors.append(colors)

    pts = np.vstack(global_pts).astype(np.float64).reshape(-1, 3)[::10]
    colors = np.vstack(global_colors).reshape(-1, 3)[::10]
    point_cloud(pts, colors)


run_dense_sfm('data/IMG_1613.mov')
