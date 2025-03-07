import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import get_custom_objects

# Register the swish activation function
def swish(x):
    return tf.nn.swish(x)

get_custom_objects().update({'swish': swish})

# Register the FixedDropout layer
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

get_custom_objects().update({'FixedDropout': FixedDropout})

app = Flask(__name__)

# Load the model
MODEL_PATH = r"D:\Leaf classifier\website\improved_leaf_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the file temporarily
    uploads_dir = os.path.join('static', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Return results
    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence,
        'image_url': file_path
    })

if __name__ == '__main__':
    app.run(debug=True)