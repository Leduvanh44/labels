import cv2
import os
from utils.read_number import blood_glucose_meter, temp_meter, sphygmomanometer
from flask import Flask, jsonify, request
import werkzeug


# image_path = 'original_img_file/74831ccc43e390bdc9f21.jpg'
# print(sphygmomanometer(image_path))


app = Flask(__name__) #intance of our flask application

@app.route ('/blood_pressure', methods=["POST"])
def uploadBloodPressureData():
    
    if(request.method =="POST"):
        imagefile=request.files['image']
        filename= werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./original_img_file/"+filename)
        filePath = f'original_img_file/{filename}'
        # sphygmomanometer(filePath)
        try:
             num = sphygmomanometer(filePath)
             sys = num[0]
             dia = num[1]
             pulse = num[2]
             print(num)
             return jsonify({"message":'Xử lý thành công',
                             "sys":sys,"dia":dia,"pulse":pulse
                         }) 
        except:
            return jsonify({"message":'Lỗi',
                        })
        

@app.route ('/temperature', methods=["POST"])
def uploadTemperatureData():
    
    if(request.method =="POST"):
        imagefile=request.files['image']
        filename= werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./original_img_file/"+filename)
        filePath = f'original_img_file/{filename}'
        # sphygmomanometer(filePath)
        try:
             temperature = temp_meter(filePath)
             print(temperature)
             return jsonify({"message":'Xử lý thành công',
                             "temperature":temperature
                         }) 
        except:
            return jsonify({"message":'Lỗi',
                        })
@app.route ('/blood_glucose', methods=["POST"])
def uploadBloodGlucoseData():
    
    if(request.method =="POST"):
        imagefile=request.files['image']
        filename= werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./original_img_file/"+filename)
        filePath = f'original_img_file/{filename}'
        # sphygmomanometer(filePath)
        try:
             glucose = blood_glucose_meter(filePath)
             print(glucose)
             return jsonify({"message":'Xử lý thành công',
                             "glucose":glucose
                         }) 
        except:
            return jsonify({"message":'Lỗi',
                        })       

if __name__ == "__main__":
    app.run(debug = True,port=4040) #debug will allow changes without shutting down the server