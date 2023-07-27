from utils.read_num_roi import read_num_roi
from utils.roi_bp import roi_bp
import cv2
import os
#performing flask imports
from flask import Flask,jsonify,request
import werkzeug

def read_number (path):
    image_path = path
    image_name_with_extension = os.path.basename(image_path)
    image_name = os.path.splitext(image_name_with_extension)[0]
    roi = cv2.imread(path)
    roi = cv2.resize(roi, (296, 385))
    cv2.imshow('roi', roi)
    roi_1 = roi_bp.crop_image(roi, 100, 25, 287, 135)
    roi_2 = roi_bp.crop_image(roi, 128, 157, 285, 286)
    roi_3 = roi_bp.crop_image(roi, 190, 286, 286, 376)
    cv2.imwrite(f'roi_num/{image_name}_roi1.png', roi_1)
    cv2.imwrite(f'roi_num/{image_name}_roi2.png', roi_2)
    cv2.imwrite(f'roi_num/{image_name}_roi3.png', roi_3)
   # roi = roi_bp.roi_blood_pressure(image_path)
    digit = read_num_roi.read_num_roi(roi_1)
    num_roi_1 = digit[0] * 100 + digit[1] * 10 + digit[2]
    digit = read_num_roi.read_num_roi(roi_2)
    num_roi_2 = digit[0] * 10 + digit[1]
    # digit_3 = cv2.imread(f'roi_num/{image_name}_roi3.png')
    digit_31 = roi_bp.crop_image(roi_3, 10, 0, 48, 90)
    digit_32 = roi_bp.crop_image(roi_3, 48, 0, 96, 90)
    digit1 = read_num_roi.read_num_roi(digit_31)
    digit2 = read_num_roi.read_num_roi(digit_32)
    num_roi_3 = digit1[0] * 10 + digit2[0]
    print(num_roi_1, num_roi_2, num_roi_3)
    #return [num_roi_1, num_roi_2,num_roi_3]
read_number('file_img/1690446442238.jpg')

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

