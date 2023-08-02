import cv2
import os
from utils.read_number import blood_glucose_meter, temp_meter, sphygmomanometer
from flask import Flask, jsonify, request
import werkzeug

# image_path = 'original_img_file/0c22d6ffe42f37716e3e7.jpg'
# print(blood_glucose_meter(image_path))
#
# image_path = 'original_img_file/642a3894fe912dcf74802.jpg'
# print(temp_meter(image_path))
#
image_path = 'original_img_file/74831ccc43e390bdc9f21.jpg'
print(sphygmomanometer(image_path))

app = Flask(__name__) #intance of our flask application

#Route '/' to facilitate get request from our flutter app
#@app.route('/', methods = ['GET'])
#def index():
#    return jsonify({'greetings' : 'Hi! this is python'}) #returning key-value pair in json format



@app.route ('/upload', methods=["POST"])
def upload():
    sys = "Image uploaded successfully"
    if(request.method =="POST"):
        imagefile=request.files['image']
        filename= werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./file_img/"+filename)
        filePath = f'file_img/{filename}'
        sphygmomanometer(filePath)
        ''' try:
             read_number(filePath)
        except:
            print('lỗi')'''
        return jsonify({"message":sys,
                         "id":1})
if __name__ == "__main__":
    app.run(debug = True,port=4040) #debug will allow changes without shutting down the server