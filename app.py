from flask import Flask, render_template,request,redirect,url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import numpy as np
import argparse
import cv2
import urllib.request
import time


app = Flask("__name__",template_folder='templates')
#img = 0
#img = "/images"
d = ''
f = ''
dataa = {}
img = {}
@app.route("/", methods= ['GET','POST'])
def home():

    if request.method == 'POST':
        #img = request.files['image']
        #d = img.filename
        #f = request.form['fname']
        #dataa['fname'] = f
        g = request.files['image']
        img['image'] = g
        i = Image.open(img['image'])
        #p = '/images' + dataa['fname']
        #i.save('/images/' + dataa['fname'])
        #i.save(dataa['fname'])
        i.save('static/uncolored.jpg')


        #Work on this
    

        #Load Model
        Dir = r""
        Prototext = os.path.join(Dir, r"model/colorization_deploy_v2.prototxt")
        Points = os.path.join(Dir, r"model/pts_in_hull.npy")
        Model = os.path.join(Dir, r"model/colorization_release_v2.caffemodel")
        net = cv2.dnn.readNetFromCaffe(Prototext,Model)
        pts = np.load(Points)

        #Load centers for ab channel quantization Used for rebaancing
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2,313,1,1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype ="float32")]


        #Load image
        image = cv2.imread('static/uncolored.jpg')
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab,(224,224))
        L = cv2.split(resized)[0]
        L -= 50

        #Colorize
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1,2,0))
        ab = cv2.resize(ab,(image.shape[1],image.shape[0]))

        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized,0,1)

        colorized = (255 * colorized).astype("uint8")

        #cv2.imshow("Original", image)
        #cv2.imshow("Colorized", colorized)
        #cv2.waitKey(0)

        #colorized.save("colorPic.jpg")
        print(type(colorized))
        print(colorized)

        data = Image.fromarray(colorized)
        data.save('static/color.jpg')
        #time.sleep(5)






        
        return redirect(url_for('output'))
    else:    
        return render_template('index.html')  

@app.route('/output')
def output():
    time.sleep(5)
    return render_template('output.html')

if __name__ == "__main__":
    app.run(debug=True,port="80")