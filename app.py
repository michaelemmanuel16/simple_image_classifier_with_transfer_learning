import numpy as np
import os
import glob
import re
import sys
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_prediction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, request, url_for, render_template
from werkzeug.utils import secure_filename


# create flask app
app = Flask(__name__)
model_path = 'vgg19.h5'

# load model
model=load_model(model_path)
model.make_predict_function()


# prediction function
def model_predict(img_path, model):
  img = image.load_img(img_path, target_size=(224,224))
  
  # preprocess the image
  x=image.img_to_array(img)
  
  x=np.expand_dims(x, axis=0)
  
  x.preprocess_input(x)
  
  preds = model.predict(x)
  return preds

@app.route('/', method=['GET'])
def index():
  return render_template('index.html')


# upload image
@app.route('/predict', method=['GET', 'POST'])
def upload():
  if request.method=='POST':
    # Get file from post
    f = request.files('file')
    # save file to uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads',secure_filename(f.filename))
    f.save(file_path)
    # make prediction
    pred=model_predict(file_path, model) 
    pred_class = decode_prediction(pred, top=1) # Decode image
    result = str(pred_class[0][0][1]) # convert to string
    return result
  return None
    
    

if __name__ =='__main__':
  app.run(debug=True)