from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib import pyplot as plt
def crop_border(pil_image, percentage):

    width, height = pil_image.size
    border_width = int(width * percentage / 100)
    border_height = int(height * percentage / 100)

    cropped_pil_image = pil_image.crop((border_width, border_height, width - border_width, height - border_height))
    return cropped_pil_image

def cv2_to_pillow(cv2_image):
    pillow_image = Image.fromarray(cv2_image)
    return pillow_image

def pillow_to_cv2(pillow_image):
    cv2_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return cv2_image

def upscale_image(cv_image, scale_factor):
    new_width = int(cv_image.shape[1] * scale_factor)
    new_height = int(cv_image.shape[0] * scale_factor)
    img = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img



# input_image_path = 'inter/58e80d47435c9002c94d6-roi.png'
input_image_path = 'roi_num/58e80d47435c9002c94d6_roi1.png'
img = cv2.imread(input_image_path)
img = upscale_image(img, 5)
img = cv2.resize(img, None, None, fx=0.3, fy=0.3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3
)

cv2.imshow("Edged", edged)
cv2.waitKey(0)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
# eroded = cv2.erode(edged, kernel, iterations=1)
# cv2.imshow("Edged", eroded)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
img = cv2.dilate(edged, kernel, iterations=1)


cv2.imshow("Edged", img)
cv2.waitKey(0)
img = cv2.convertScaleAbs(img, 0.5, 2.25)
img = cv2.medianBlur(img, 5)
img = cv2_to_pillow(img)
sharpened_image = img.filter(ImageFilter.SHARPEN)
img = pillow_to_cv2(img)
plt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(plt_image)
plt.show()
cv2.imshow('result', img)
cv2.waitKey(0)
image_name_with_extension = os.path.basename(input_image_path)
image_name, _ = os.path.splitext(image_name_with_extension)
image_path = f'edit/{image_name}-edit.png'
cv2.imwrite(image_path, img)


orc = PaddleOCR(lang='en')
img_path = os.path.join('.', image_path)
result = orc.ocr(img_path)
num_list = []
try:
    i, j = 0, 0
    while i < 100:
        print(result[0][i][1][0])
        i += 1
        # if isinstance(result[0][i][1][0], str) and result[0][i][1][0].isalpha():
        #     num_list[j] = result[0][i][1][0]
        #     j +=1
except IndexError:
    pass
print(num_list)








