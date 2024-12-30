from flask import Flask, jsonify, request
from flask_cors import CORS

import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
import numpy as np
from flask import session
import uuid
import json
from datetime import datetime

from PIL import Image
import cv2
import os
import os
import io

import tensorflow as tf
from PIL import Image
import os
import PIL
import cv2
import keras
import matplotlib.pyplot as plt
UPLOAD_FOLDER = './uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
NUM_CLASSES = 13
Str_to_Int ={'Anthracnose on Cotton': 0,
 'Becterial Blight in Rice': 1,
 'Brownspot': 2,
 'Common_Rust': 3,
 'Cotton Aphid': 4,
 'Gray_Leaf_Spot': 5,
 'Healthy cotton': 6,
 'Healthy Maize': 7,
 'Healthy Wheat': 8,
 'Rice Blast': 9,
 'Tungro': 10,
 'Wheat aphid': 11,
 'Wheat Brown leaf Rust': 12}
# Reverse the dictionary
Int_to_Str = {v: k for k, v in Str_to_Int.items()}
grouped_plants = {
    'Cotton': ['Anthracnose on Cotton', 'Cotton Aphid', 'Healthy cotton'],
    'Rice': ['Becterial Blight in Rice', 'Rice Blast', 'Tungro'],
    'Maize': ['Common_Rust', 'Gray_Leaf_Spot', 'Healthy Maize'],
    'Wheat': ['Healthy Wheat', 'Wheat aphid', 'Wheat Brown leaf Rust','Brownspot']
}
disease_remedy = {
    'Anthracnose on Cotton': 'Use fungicides such as azoxystrobin or copper-based sprays and remove infected plant debris.',
    'Becterial Blight in Rice': 'Apply copper-based bactericides and use resistant rice varieties.',
    'Brownspot': 'Use fungicides containing mancozeb or carbendazim and maintain proper plant spacing to improve air circulation.',
    'Common_Rust': 'Apply fungicides like azoxystrobin or propiconazole and plant resistant maize varieties.',
    'Cotton Aphid': 'Use insecticides like imidacloprid or neem oil and introduce natural predators like ladybugs.',
    'Gray_Leaf_Spot': 'Apply fungicides like strobilurins and triazoles and avoid over-irrigation to prevent excess moisture.',
    'Healthy cotton': 'No treatment required. Maintain proper growing conditions to keep the cotton healthy.',
    'Healthy Maize': 'No treatment required. Ensure adequate nutrition and pest control for optimal health.',
    'Healthy Wheat': 'No treatment required. Monitor for pests and diseases to maintain wheat health.',
    'Rice Blast': 'Apply triazole fungicides like tricyclazole and avoid excessive nitrogen fertilization.',
    'Tungro': 'Control vector insects (green leafhoppers) using insecticides like imidacloprid and plant resistant rice varieties.',
    'Wheat aphid': 'Use insecticides such as pyrethroids or natural control methods like ladybugs.',
    'Wheat Brown leaf Rust': 'Apply fungicides like tebuconazole and use rust-resistant wheat varieties.'
}
def preprocess(IMG_SAVE_PATH):
    dataset = []
   
    try:
                imgpath=PIL.Image.open(IMG_SAVE_PATH)
                imgpath=imgpath.convert('RGB')
                img = np.asarray(imgpath)
                img = cv2.resize(img, (331,331))
                img=img/255.
                dataset.append(img)
    except FileNotFoundError:
                print('Image file not found. Skipping...')
    return dataset
model = tf.keras.models.load_model('plantdisease.h5')
model.summary()

app = Flask("Plant Disease Detector")
CORS(app)
@app.route('/')
def hello():
    message= ''
    return "plant Leaf Diases"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image to the upload folder
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    traindata=preprocess(image_path)
    xtest=np.array(traindata)
  
    Y_pred = model.predict([xtest, xtest])
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    print(Y_pred_classes)
    detected_disease=Int_to_Str[Y_pred_classes[0]]
    plant_name = None
    for plant, diseases in grouped_plants.items():
        if detected_disease in diseases:
            plant_name = plant
            break

    
    response = {
        "plant": plant_name,
        "disease":detected_disease ,
        "remedy": disease_remedy[detected_disease],
    }
    response = jsonify(response)
    print(response)
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
