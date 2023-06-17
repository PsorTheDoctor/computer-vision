import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

kelvinTable = {
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)
}


def estimateColor(temp):
    kelvins = np.array(list(kelvinTable.keys()))
    colors = np.array(list(kelvinTable.values()))

    degree = 3
    r = np.polyfit(kelvins, colors[:, 0], degree)
    g = np.polyfit(kelvins, colors[:, 1], degree)
    b = np.polyfit(kelvins, colors[:, 2], degree)

    pred_r = np.poly1d(r)
    pred_g = np.poly1d(g)
    pred_b = np.poly1d(b)

    color = [int(pred_r(temp)), int(pred_g(temp)), int(pred_b(temp))]

    for channel in range(len(color)):
        if color[channel] < 0:
            color[channel] = 0
        elif color[channel] > 255:
            color[channel] = 255

    return color


def adjustColorTemp(img, targetTemp):
    color = estimateColor(targetTemp)

    r = color[0]
    g = color[1]
    b = color[2]

    mat = [r / 255.0, 0.0, 0.0, 0.0,
           0.0, g / 255.0, 0.0, 0.0,
           0.0, 0.0, b / 255.0, 0.0]

    result = img.convert('RGB', mat)
    return result


img = Image.open('../data/sunflower.jpg')
result = adjustColorTemp(img, 3000)
plt.figure()
plt.imshow(result)
plt.axis('off')

result = adjustColorTemp(img, 20000)
plt.figure()
plt.imshow(result)
plt.axis('off')
plt.show()
