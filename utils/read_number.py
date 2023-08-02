from paddleocr import PaddleOCR
import re
from utils.read_num_roi import read_num_roi
from utils.roi_bp import roi_bp
import cv2
import os
ocr = PaddleOCR(lang='en')
def blood_glucose_meter(input_image_path):
    cv_img = cv2.imread(input_image_path)
    cv_img = cv2.resize(cv_img, None, None, fx=0.3, fy=0.3)
    blurred = cv2.GaussianBlur(cv_img, (5, 5), 10)
    cv_img = cv2.bilateralFilter(blurred, 10, sigmaColor=50, sigmaSpace=50)
    image_name_with_extension = os.path.basename(input_image_path)
    image_name, _ = os.path.splitext(image_name_with_extension)
    cv2.imwrite(f'edit/{image_name}_edit.png', cv_img)
    edit_cv_img_path = f'edit/{image_name}_edit.png'
    cv_img_path = os.path.join('', edit_cv_img_path)
    result = ocr.ocr(cv_img_path)
    os.system('cls')
    res_list = []
    try:
        i = 0
        while i < 50:
            res_list.append(result[0][i][1][0])
            i += 1
    except IndexError:
        pass
    num_list = []
    for res in res_list:
        if re.match(r'^\d+(\.\d+)?(\.\d+)?$', str(res)):
            num_list.append(res)
    if len(num_list) == 1:
        return num_list[0]
    else:
        print('Error Error')
        return num_list[-1]


def temp_meter(input_image_path):
    cv_img = cv2.imread(input_image_path)
    cv_img = cv2.resize(cv_img, None, None, fx=0.3, fy=0.3)
    cv_img = cv2.GaussianBlur(cv_img, (5, 5), 8)
    cv_img = cv2.bilateralFilter(cv_img, 8, sigmaColor=50, sigmaSpace=50)
    image_name_with_extension = os.path.basename(input_image_path)
    image_name, _ = os.path.splitext(image_name_with_extension)
    cv2.imwrite(f'edit/{image_name}_edit.png', cv_img)
    edit_cv_img_path = f'edit/{image_name}_edit.png'
    cv_img_path = os.path.join('', edit_cv_img_path)
    result = ocr.ocr(cv_img_path)
    os.system('cls')
    res_list = []
    try:
        i = 0
        while i < 50:
            res_list.append(result[0][i][1][0])
            i += 1
    except IndexError:
        pass
    num_list = []
    for res in res_list:
        if re.match(r'^\d+(\.\d+)?$', str(res)):
            num_list.append(res)
    if len(num_list) == 1:
        if int(num_list[0]) > 100:
            num_list[0] = str(float(num_list[0])/10)
        return num_list[0]
    else:
        print('Error Error')
        return num_list[-1]


def sphygmomanometer(image_path):
    image_name_with_extension = os.path.basename(image_path)
    image_name = os.path.splitext(image_name_with_extension)[0]
    roi = roi_bp.roi_blood_pressure(image_path, canny=85)
    digit = roi
    digit1_png = roi_bp.crop_image(digit, 100, 21, 153, 134)
    digit2_png = roi_bp.crop_image(digit, 153, 20, 218, 134)
    digit3_png = roi_bp.crop_image(digit, 219, 20, 283.5, 134)
    digit1 = read_num_roi.read_num_roi(digit1_png, True)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    digit3 = read_num_roi.read_num_roi(digit3_png)
    num_roi_1 = digit1[0] * 100 + digit2[0] * 10 + digit3[0]
    digit1_png = roi_bp.crop_image(digit, 100, 174, 153, 287)
    digit2_png = roi_bp.crop_image(digit, 153, 174, 218, 287)
    digit3_png = roi_bp.crop_image(digit, 219, 174, 283.5, 287)
    digit1 = read_num_roi.read_num_roi(digit1_png, True)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    digit3 = read_num_roi.read_num_roi(digit3_png)
    num_roi_2 = digit1[0]*100 + digit2[0] * 10 + digit3[0]
    digit1_png = roi_bp.crop_image(digit, 190, 288, 246.8, 375)
    digit2_png = roi_bp.crop_image(digit, 247, 288, 298, 375)
    digit1 = read_num_roi.read_num_roi(digit1_png)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    num_roi_3 = digit1[0] * 10 + digit2[0]
    return [num_roi_1, num_roi_2, num_roi_3]








