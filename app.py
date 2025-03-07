from flask import Flask, request, jsonify, render_template  # Add render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the FixedDropout layer
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

# Define custom F1Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "improvedd_leaf_classifier_with_grapes.keras"
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'FixedDropout': FixedDropout, 'F1Score': F1Score}
)

# Define class names (replace with your actual class names)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy"
]

# Preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match training data
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html file

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    img_path = "temp_image.jpg"
    file.save(img_path)

    # Preprocess the image and make a prediction
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    # Clean up the temporary file
    os.remove(img_path)

    # Return the result
    return jsonify({
        "class": predicted_class,
        "confidence": confidence
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)