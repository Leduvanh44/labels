import cv2
import numpy as np
import time
import os
from utils.roi_bp.roi_bp import find_closest_tuple
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
DIGITSDICT_tuple = [
    (1, 1, 1, 1, 1, 1, 0),
    (0, 1, 1, 0, 0, 0, 0),
    (1, 1, 0, 1, 1, 0, 1),
    (1, 1, 1, 1, 0, 0, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 1, 0, 1, 1),
    (1, 0, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 0, 1, 1),
]
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}
def temp(image_path, flash):
    roi = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # num_channels = cv_img.shape[2]
    # if num_channels > 1:
    #     try:
    #         roi = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    #     except:
    #         pass
    roi = cv2.GaussianBlur(cv2.resize(roi, (350, 170)), (5, 5), 1)
    roi = cv2.bilateralFilter(roi, 1, sigmaColor=10, sigmaSpace=75)
    roi = cv2.resize(roi, None, None, fx=2.2, fy=1.5)
    if flash == 'true':
        print('flash on')
        edged = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=3)
    else:
        print('flash off')
        edged = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        (_, _, w, h) = cv2.boundingRect(contour)
        if (cv2.contourArea(contour) > 380) & (w < 140) & (h < 140):
            filtered_contours += [contour]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ##################################################
    dilated = edged
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (4, 8))
    dilated = cv2.dilate(dilated, kernel, iterations=2)
    erosion_kernel = np.ones((2, 3), np.uint8)
    eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
    ##################################################
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_cnts = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # print(w * h)
        if w * h > 5000:
            digits_cnts += [cnt]
    sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])[:3]
    digits = []
    for i, cnt in enumerate(sorted_digits):
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = eroded[y: y + h, x: x + w]
        qW, qH = int(w * 0.25), int(h * 0.15)
        fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW - qW, halfH), (w - qW, h)),  # c (lower right)
            ((0, h - qH), (w, h)),  # d (lower bar)
            ((0, halfH), (qW, h)),  # e (lower left)
            ((0, 0), (qW, halfH)),  # f (upper left)
            # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
            (
                (0 + fractionW, halfH - fractionH),
                (w - fractionW, halfH + fractionH),
            ),  # center
        ]
        on = [0] * 7
        for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
            region = roi[p1y:p2y, p1x:p2x]
            if np.sum(region == 255) > region.size * 0.5:
                on[i] = 1
        if on not in DIGITSDICT_tuple:
            closest_tuple = find_closest_tuple(on, DIGITSDICT_tuple)
            index = DIGITSDICT_tuple.index(closest_tuple)
            on = DIGITSDICT_tuple[index]
        digit = DIGITSDICT[tuple(on)]
        if digit != 1:
            if w < 90:
                digit = 1
        digits += [digit]
    # print(digits)
    print('--------------------------------------------++')
    if len(digits) == 3:
        digits = str(float(digits[0] * 10 + digits[1] + digits[2] / 10))
    else:
        digits = 'Error'
    return digits

if __name__ == '__main__':
    folder_path = 'ori_img_5'
    images = os.listdir(folder_path)
    print(images)
    for image in images:
        file_path = os.path.join(folder_path, image)
        if os.path.isfile(file_path) and image.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(image)[0]
            print(name)
            image_path = f'{folder_path}/{image}'
            temp(image_path)
            # cv2.imshow(f'{name}', roi)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('a'):
            #         cv2.destroyAllWindows()
            #         break
