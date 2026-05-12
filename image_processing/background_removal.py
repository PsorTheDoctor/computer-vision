from rembg import remove
from PIL import Image

img = Image.open('../data/adam.jpg')
result = remove(
    img,
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=1
)
result.save('result.png')
