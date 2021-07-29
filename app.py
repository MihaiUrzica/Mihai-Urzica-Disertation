import os

from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename


import numpy as np

from tensorflow.keras.preprocessing import image
from keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
from os import path

import tensorflow.compat.v1 as td
td.disable_v2_behavior()

X = "Negativ"
Y = "Pozitiv"

sampleX='static/pneumonie.jpg'
sampleY='static/non_pneumonie.jpg'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ML_MODEL_FILENAME = 'model1.h5'



app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')


def load_model_from_file():
    #Set up the machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    file_name = ML_MODEL_FILENAME
    myModel = load_model(file_name)
    myGraph = tf.compat.v1.get_default_graph()
    return (mySession,myModel,myGraph)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY)
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename,target_size=(500,500), color_mode="grayscale")
    save_image = test_image
    try: 
        # test_image = tf.image.rgb_to_grayscale(test_image)
        print("Image has been converted to black and white")
    except: 
        print("Image has NOT been converted to black and white")
    try: 
        test_image = image.img_to_array(test_image)
        print("Succes in converting")
    except: 
        print("Error in converting")

    test_image = np.expand_dims(test_image, axis=0)
    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)[0][0]
        print(result)
        image_src = "/"+UPLOAD_FOLDER +"/"+filename
        if result < 0.5 :
            answer = "<div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>Rezultat:"+X+" "+"</h4></div><div class='col'></div><div class='w-100'></div>"  
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>Rezultat:"+Y+" "+"</h4></div><div class='w-100'></div>"     
        results.append(answer)
        return render_template('index.html',myX=X,myY=Y,mySampleX=sampleX,mySampleY=sampleY,len=len(results),results=results)

def main():
    (mySession,myModel,myGraph) = load_model_from_file()
    
    app.config['SECRET_KEY'] = 'super secret key'

    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16MB upload limit
    app.run()

results = []

main()