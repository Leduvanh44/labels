from utils.read_num_roi import read_num_roi
from utils.roi_bp import roi_bp
import cv2
import os

image_path = 'file_img/58e80d47435c9002c94d6.jpg'
image_name_with_extension = os.path.basename(image_path)
image_name = os.path.splitext(image_name_with_extension)[0]
print(image_name)
roi = roi_bp.roi_blood_pressure(image_path)
cv2.imshow('Main Roi', roi)
digit = read_num_roi.read_num(f'roi_num/{image_name}_roi1.png')
num_roi_1 = digit[0] * 100 + digit[1] * 10 + digit[2]
digit = read_num_roi.read_num(f'roi_num/{image_name}_roi2.png')
num_roi_2 = digit[0] * 10 + digit[1]
digit_3 = cv2.imread(f'roi_num/{image_name}_roi3.png')
digit_31 = roi_bp.crop_image(digit_3, 10, 0, 48, 90)
digit_32 = roi_bp.crop_image(digit_3, 48, 0, 96, 90)
digit1 = read_num_roi.read_num_roi(digit_31)
digit2 = read_num_roi.read_num_roi(digit_32)
num_roi_3 = digit1[0] * 10 + digit2
print(num_roi_1, num_roi_2, num_roi_3)