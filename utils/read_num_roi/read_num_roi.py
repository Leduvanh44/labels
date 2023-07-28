import cv2
import numpy as np
import math
from PIL import Image
def cv2_to_pillow(cv2_image):
    pillow_image = Image.fromarray(cv2_image)
    return pillow_image

def pillow_to_cv2(pillow_image):
    cv2_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return cv2_image

def crop_percentage_border(pil_image, percentage):
    width, height = pil_image.size
    border_width = int(width * percentage / 100)
    border_height = int(height * percentage / 100)
    cropped_pil_image = pil_image.crop((border_width, border_height, width - border_width, height - border_height))
    return cropped_pil_image

def euclidean_distance(tuple1, tuple2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(tuple1, tuple2)))

def find_closest_tuple(target_list, tuple_list):
    closest_tuple = None
    closest_distance = float('inf')

    for tuple_item in tuple_list:
        distance = euclidean_distance(target_list, tuple_item)
        if distance < closest_distance:
            closest_tuple = tuple_item
            closest_distance = distance
    return closest_tuple

def replace_tuple_if_not_present(target_tuple, tuple_list):
    try:
        index = tuple_list.index(target_tuple)
    except ValueError:
        closest_tuple = find_closest_tuple(target_tuple, tuple_list)
        index = tuple_list.index(closest_tuple)
    tuple_list[index] = target_tuple

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
    (0, 1, 1, 1, 1, 0, 1),
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
    (0, 1, 1, 1, 1, 0, 1): 9,
}

def read_num_roi(roi_color, num_roi=False, ratio=0.001):
    roi_color = cv2.resize(roi_color, None, None, fx=2, fy=1)
    roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    RATIO = roi.shape[0] * ratio
    roi = cv2.bilateralFilter(roi, 5, 30, 60)
    trimmed = roi[int(RATIO):, int(RATIO): roi.shape[1] - int(RATIO)]
    roi_color = roi_color[int(RATIO):, int(RATIO): roi.shape[1] - int(RATIO)]
    # cv2.imshow("Blurred and Trimmed", trimmed)
    # cv2.waitKey(0)

    edged = cv2.adaptiveThreshold(
        trimmed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3.5
    )
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)

    # edged = pillow_to_cv2(crop_percentage_border(cv2_to_pillow(edged), 7))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    # cv2.imshow("Dilated", dilated)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilated = cv2.dilate(dilated, kernel, iterations=1)

    # cv2.imshow("Dilated x2", dilated)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1), )
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # cv2.imshow("Eroded", eroded)
    # cv2.waitKey(0)

    cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_cnts = []

    canvas = trimmed.copy()
    cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
    # cv2.imshow("All Contours", canvas)
    # cv2.waitKey(0)

    canvas = trimmed.copy()
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h > 20:
            digits_cnts += [cnt]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.drawContours(canvas, cnt, 0, (255, 255, 255), 1)
            # cv2.imshow("Digit Contours", canvas)
            # cv2.waitKey(0)
    # print(f"No. of Digit Contours: {len(digits_cnts)}")
    # cv2.imshow("Digit Contours", canvas)
    # cv2.waitKey(0)
    sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])
    canvas = trimmed.copy()
    for i, cnt in enumerate(sorted_digits):
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.putText(canvas, str(i), (x, y - 3), FONT, 0.3, (0, 0, 0), 1)
    # cv2.imshow("All Contours sorted", canvas)
    # cv2.waitKey(0)
    digits = []
    canvas = roi_color.copy()
    for cnt in sorted_digits:
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = eroded[y: y + h, x: x + w]
        print(f"W:{w}, H:{h}")
        # convenience units
        qW, qH = int(w * 0.25), int(h * 0.15)
        fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)
        # ưiki
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW, halfH), (w, h)),  # c (lower right)
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
            # print(
            #     f"{i}: Sum of 1: {np.sum(region == 255)}, Sum of 0: {np.sum(region == 0)}, Shape: {region.shape}, Size: {region.size}"
            # )
            if np.sum(region == 255) > region.size * 0.5:
                on[i] = 1
            # print(f"State of ON: {on}")
        if on not in DIGITSDICT_tuple:
            closest_tuple = find_closest_tuple(on, DIGITSDICT_tuple)
            index = DIGITSDICT_tuple.index(closest_tuple)
            on = DIGITSDICT_tuple[index]
        digit = DIGITSDICT[tuple(on)]
        if digit == 2:
            if w < 50:
                digit = 1
        if num_roi == True:
            if w < 20:
                digit = 0
            else:
                digit = 1
        # print(f"Digit is: {digit}")
        digits += [digit]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), CYAN, 1)
        cv2.putText(canvas, str(digit), (x - 5, y + 6), FONT, 0.3, (0, 0, 0), 1)
        # cv2.imshow("Digit", canvas)
        # cv2.waitKey(0)

    print(f"Digits in img are: {digits}")
    if len(digits) == 0:
        digits = [0]
    return digits