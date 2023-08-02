import cv2
import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
PURPLE = (75, 0, 130)
YELLOW = (0, 255, 255)
THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
def increase_contrast(image, factor):
    img_array = np.array(image)
    mean_value = np.mean(img_array)
    contrasted_array = (img_array - mean_value) * factor + mean_value
    contrasted_array = np.clip(contrasted_array, 0, 255)
    contrasted_image = Image.fromarray(contrasted_array.astype('uint8'))
    return contrasted_image

def crop_image(image, left, top, right, bottom):
    image = cv2_to_pillow(image)
    cropped_image = image.crop((left, top, right, bottom))
    return pillow_to_cv2(cropped_image)

def cv2_to_pillow(cv2_image):
    pillow_image = Image.fromarray(cv2_image)
    return pillow_image

def pillow_to_cv2(pillow_image):
    cv2_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return cv2_image



def reduce_blur(pil_image, sigma):
    img_array = np.array(pil_image)
    blurred_array = gaussian_filter(img_array, sigma=sigma)
    blurred_pil_image = Image.fromarray(blurred_array.astype('uint8'))
    return blurred_pil_image


def roi_blood_pressure(img_path):
    peri_pre = 0
    x_cur, y_cur, w_cur, h_cur, area_cur, peri_cur = 0, 0, 0, 0, 0, 0
    img_color = cv2.imread(img_path)
    img_color = cv2.resize(img_color, (480, 640))
    img_color_c = img_color.copy()
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    blurred = cv2.bilateralFilter(blurred, 5, sigmaColor=50, sigmaSpace=50)
    edged = cv2.Canny(blurred, 130, 150, 255)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:100]
    cv2.drawContours(img_color, cnts, 0, PURPLE, THICKNESS)
    for i in range(len(cnts)):
        cv2.drawContours(img_color, cnts, i, PURPLE, THICKNESS)
        # print(f"ContourArea:{cv2.contourArea(cnts[i])}")
        x, y, w, h = cv2.boundingRect(cnts[i])
        cv2.rectangle(img_color, (x, y), (x + w, y + h), YELLOW, THICKNESS)
        area = round(cv2.contourArea(cnts[i]), 1)
        peri = round(cv2.arcLength(cnts[i], closed=True), 1)
        if peri >= peri_pre:
            peri_pre = peri
            x_cur, y_cur, w_cur, h_cur, area_cur, peri_cur = x, y, w, h, area, peri
        # print(f"ContourArea:{area}, Peri: {peri}")
        # cv2.putText(img_color, "Area:" + str(area), (x, y - 15), FONT, 0.4, PURPLE, 1)
        # cv2.putText(img_color, "Perimeter:" + str(peri), (x, y - 5), FONT, 0.4, PURPLE, 1)

    # print(x_cur, y_cur, w_cur, h_cur, area_cur, peri_cur)
    roi = img_color_c[y_cur: y_cur + h_cur, x_cur: x_cur + w_cur]
    roi = cv2.resize(roi, (296, 385))
    roi = pillow_to_cv2(increase_contrast(cv2_to_pillow(roi), 2))
    image_name_with_extension = os.path.basename(img_path)
    image_name, _ = os.path.splitext(image_name_with_extension)

    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)

    roi_path = image_name + '-roi.png'
    cv2.imwrite(f'inter/{roi_path}', roi)
    roi_1 = crop_image(roi, 100, 25, 287, 135)
    roi_2 = crop_image(roi, 128, 157, 285, 286)
    roi_3 = crop_image(roi, 190, 286, 286, 376)
    # cv2.imshow('roi_1', roi_1)
    # cv2.imshow('roi_2', roi_2)
    # cv2.imshow('roi_3', roi_3)
    # cv2.waitKey(0)

    cv2.imwrite(f'roi_num/{image_name}_roi1.png', roi_1)
    cv2.imwrite(f'roi_num/{image_name}_roi2.png', roi_2)
    cv2.imwrite(f'roi_num/{image_name}_roi3.png', roi_3)
    return roi



