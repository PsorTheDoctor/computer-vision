import cv2
import random
import time


def selectiveSearch(method):
    img = cv2.imread('../data/sunflower.jpg')
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if method == 'fast':
        ss.switchToSelectiveSearchFast()
    elif method == 'quality':
        ss.switchToSelectiveSearchQuality()

    results = ss.process()

    for i in range(0, len(results), 100):
        output = img.copy()
        for (x, y, w, h) in results[i:i + 100]:
            color = [random.randint(0, 255) for j in range(0, 3)]
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

        cv2.imshow(method, output)
        time.sleep(0.1)
        if cv2.waitKey(50) == 27:
            break


selectiveSearch('fast')
selectiveSearch('quality')
