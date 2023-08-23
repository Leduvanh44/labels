import os
from utils.temp import temp
from utils.gluco import roi_glu
from utils.press import roi_press
from flask import Flask, jsonify, request
import werkzeug
import subprocess

# create vir env
# subprocess.run(['pip', 'install', 'virtualenv'])
# venv_name = 'myenv'
# subprocess.run(['virtualenv', venv_name])
# requirements_file = 'requirements.txt'
# activate_command = f'{venv_name}/bin/activate'
# subprocess.run(activate_command, shell=True)
# install_command = f'pip install -r {requirements_file}'
# subprocess.run(install_command, shell=True)
# deactivate_command = 'deactivate'
# subprocess.run(deactivate_command, shell=True)



app = Flask(__name__) 
@app.route ('/blood_pressure', methods=["POST"])
def uploadBloodPressureData():
    if(request.method =="POST"):
        imagefile = request.files['image']
        flashOn = request.form['flashOn']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        name = os.path.splitext(filename)[0]
        filename = f'{name}.{flashOn}.jpg'
        imagefile.save("./original_img_file/"+filename)
        filePath = f'original_img_file/{filename}'
        try:
             num = roi_press(filePath)
             sys = num[0]
             dia = num[1]
             pulse = num[2]
             print(num)
             # delete file after giving number in file
             # if os.path.exists(filePath):
             #     os.remove(filePath)
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
        flashOn = request.form['flashOn']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./original_img_file/"+filename)
        name = os.path.splitext(filename)[0]
        filePath = f'original_img_file/{name}.{flashOn}.jpg'
        try:
             temperature = temp(filePath)
             if temperature == 'Error':
                 # if os.path.exists(filePath):
                 #     os.remove(filePath)
                 return jsonify({"message": 'Lỗi',
                                 })
             # if os.path.exists(filePath):
             #     os.remove(filePath)
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
        flashOn = request.form['flashOn']
        filename= werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./original_img_file/"+filename)
        name = os.path.splitext(filename)[0]
        filePath = f'original_img_file/{name}.{flashOn}.jpg'
        try:
             glucose, _ = roi_glu(filePath)
             print(glucose)
             # if os.path.exists(filePath):
             #     os.remove(filePath)
             return jsonify({"message":'Xử lý thành công',
                             "glucose":glucose
                         }) 
        except:
            return jsonify({"message":'Lỗi',
                        })       

if __name__ == "__main__":
    app.run(debug = True,port=4040) #debug will allow changes without shutting down the server