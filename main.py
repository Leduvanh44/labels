from utils.read_num_roi import read_num_roi
from utils.roi_bp import roi_bp
import cv2
import os
#performing flask imports
from flask import Flask,jsonify,request
import werkzeug

from utils.read_num_roi import read_num_roi
from utils.roi_bp import roi_bp
import cv2
import os
from flask import Flask,jsonify,request
import werkzeug
def read_number(image_path):
    image_name_with_extension = os.path.basename(image_path)
    image_name = os.path.splitext(image_name_with_extension)[0]

    roi = roi_bp.roi_blood_pressure(image_path, canny=85)
    cv2.imshow('roi', roi)
    cv2.waitKey(0)
    # roi_1 = roi_bp.crop_image(roi, 100, 21, 287, 135)
    # roi_2 = roi_bp.crop_image(roi, 100, 157, 285, 286)
    # roi_3 = roi_bp.crop_image(roi, 190, 286, 286, 376)
    # cv2.imshow('roi_1', roi_1)
    # cv2.imshow('roi_2', roi_2)
    # cv2.imshow('roi_3', roi_3)
    # cv2.waitKey(0)
    #
    # cv2.imwrite(f'roi_num/{image_name}_roi1.png', roi_1)
    # cv2.imwrite(f'roi_num/{image_name}_roi2.png', roi_2)
    # cv2.imwrite(f'roi_num/{image_name}_roi3.png', roi_3)

    # digit = read_num_roi.read_num(f'roi_num/{image_name}_roi1.png')
    digit = roi
    digit1_png = roi_bp.crop_image(digit, 100, 21, 153, 134)
    digit2_png = roi_bp.crop_image(digit, 153, 20, 218, 134)
    digit3_png = roi_bp.crop_image(digit, 219, 20, 283.5, 134)

    digit1 = read_num_roi.read_num_roi(digit1_png, True)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    digit3 = read_num_roi.read_num_roi(digit3_png)
    num_roi_1 = digit1[0] * 100 + digit2[0] * 10 + digit3[0]

    # digit = read_num_roi.read_num(f'roi_num/{image_name}_roi2.png')
    digit1_png = roi_bp.crop_image(digit, 100, 174, 153, 287)
    digit2_png = roi_bp.crop_image(digit, 153, 174, 218, 287)
    digit3_png = roi_bp.crop_image(digit, 219, 174, 283.5, 287)
    digit1 = read_num_roi.read_num_roi(digit1_png, True)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    digit3 = read_num_roi.read_num_roi(digit3_png)
    num_roi_2 = digit1[0]*100 + digit2[0] * 10 + digit3[0]


    # digit_3 = cv2.imread(f'roi_num/{image_name}_roi3.png')
    digit1_png = roi_bp.crop_image(digit, 190, 288, 246.8, 375)
    digit2_png = roi_bp.crop_image(digit, 247, 288, 298, 375)
    digit1 = read_num_roi.read_num_roi(digit1_png)
    digit2 = read_num_roi.read_num_roi(digit2_png)
    num_roi_3 = digit1[0] * 10 + digit2[0]

    return [num_roi_1, num_roi_2, num_roi_3]

image_path = 'file_img/64d763b23c9defc3b68c2.jpg'
num = read_number(image_path)
print(num)

# app = Flask(__name__) #intance of our flask application 

# #Route '/' to facilitate get request from our flutter app
# #@app.route('/', methods = ['GET'])
# #def index():
# #    return jsonify({'greetings' : 'Hi! this is python'}) #returning key-value pair in json format



# @app.route ('/upload', methods=["POST"])
# def upload():
#     sys = "Image uploaded successfully"
#     if(request.method =="POST"):
#         imagefile=request.files['image']
#         filename= werkzeug.utils.secure_filename(imagefile.filename)
#         imagefile.save("./file_img/"+filename)
#         filePath = f'file_img/{filename}'
#         read_number(filePath)
#         ''' try:
#              read_number(filePath)
#         except:
#             print('lỗi')'''
#         return jsonify({"message":sys,
#                          "id":1})                                             
# if __name__ == "__main__":
#     app.run(debug = True,port=4000) #debug will allow changes without shutting down the server 

