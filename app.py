from __future__ import annotations
import os
from typing import Any

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.utils import img_to_array, load_img

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.keras')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-digit', methods=['POST'])
def predict_digit() -> Any:
    """Handle digit prediction requests."""
    num = request.form.get('num')
    if not num or not num.isdigit():
        return jsonify({"error": "Please enter a valid number"}), 400
    
    try:
        num = int(num)
        if not 0 <= num <= 9:
            return jsonify({"error": "Please enter a number between 0 and 9"}), 400
            
        # Here you can add your specific digit prediction logic
        return jsonify({
            "prediction": num,
            "confidence": 0.95
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-image', methods=['POST'])
def predict_image() -> Any:
    """Handle image prediction requests."""
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = load_img(filepath, target_size=(28, 28), color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = 255 - img_array
        img_array = img_array.reshape(1, 28*28)
        
        # Predict
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        
        return jsonify({
            "prediction": predicted_class,
            "image_path": filepath
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)