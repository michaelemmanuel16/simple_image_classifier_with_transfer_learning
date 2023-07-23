from flask import Flask, redirect, request, url_for, render_template
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_prediction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# create flask app
app = Flask(__name__)
model_path = 'vgg19.h5'

# load model
model=load_model(model_path)
model.make_predict_function()


if __name__ =='__main__':
  app.run(debug=True)