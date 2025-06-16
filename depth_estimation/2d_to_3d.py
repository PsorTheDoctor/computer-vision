import numpy as np
import cv2
import torch

model_type = 'DPT_Large'
model = torch.hub.load('intel-isl/MiDaS', model_type)
model.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.dpt_transform if 'DPT' in model_type else transforms.small_transform

img = cv2.imread('../data/adam.jpg')
img = cv2.resize(img, (500, 500))
batch = transform(img)

with torch.no_grad():
    pred = model(batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=img.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze()

depth = pred.numpy()
depth = (depth - depth.min()) / (depth.max() -depth.min() + 1e-6)

h, w = depth.shape
baseline = 0.05 * w  # should be in range between 0.02 and 0.08
left_img = np.zeros_like(img)
right_img = np.zeros_like(img)
left_mask = np.zeros((h, w), dtype=np.uint8)
right_mask = np.zeros((h, w), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        shift = int((1 - depth[y, x]) * baseline)
        x_left = x + shift
        if 0 <= x_left < w:
            left_img[y, x] = img[y, x_left]
        else:
            left_mask[y, x] = 255

        x_right = x - shift
        if 0 <= x_right < w:
            right_img[y, x] = img[y, x_right]
        else:
            right_mask[y, x] = 255

left_inpaint = cv2.inpaint(left_img, left_mask, 3, cv2.INPAINT_TELEA)
right_inpaint = cv2.inpaint(right_img, right_mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Left', left_inpaint)
cv2.imshow('Right', left_inpaint)
cv2.waitKey(0)
cv2.destroyAllWindows()

stereo_sbs = np.hstack((left_inpaint, right_inpaint))
cv2.imwrite('adam_3d_sbs.jpg', stereo_sbs)
