import cv2
import numpy as np
import open3d as o3d


def load_video(path, max_frames=100):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
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


def get_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, pts1, pts2


def triangulate(P1, P2, pts1, pts2):
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    mask = np.isfinite(pts_3d).all(axis=1)
    return pts_3d[mask]


def point_cloud(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


def run_sfm(path):
    frames = load_video(path)
    K = get_intrinsics(frames[0])
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))
    R_prev = R_global.copy()
    t_prev = t_global.copy()
    pts = []

    for i in range(0, len(frames) - 5, 5):
        pts1, pts2 = detect_and_match(frames[i], frames[i + 5])
        R, t, pts1, pts2 = get_pose(pts1, pts2, K)
        t_global = t_global + R_global @ t
        R_global = R @ R_global

        # Projection matrices
        P1 = K @ np.hstack((R_prev, t_prev))
        P2 = K @ np.hstack((R_global, t_global))

        pts.extend(triangulate(P1, P2, pts1, pts2))
        R_prev = R_global.copy()
        t_prev = t_global.copy()

    point_cloud(np.array(pts))


run_sfm('data/IMG_1613.mov')
