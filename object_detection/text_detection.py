import easyocr
from PIL import Image, ImageDraw

filename = '../data/fans.jpg'
img = Image.open(filename)

reader = easyocr.Reader(['da', 'en'])
bounds = reader.readtext(filename)
print(bounds)


def draw_boxes(img, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(img)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return img


draw_boxes(img, bounds).show()
