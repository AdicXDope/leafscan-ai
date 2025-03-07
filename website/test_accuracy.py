import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the model
model = tf.keras.models.load_model('improved_leaf_classifier.keras')

# Data generator for validation
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    r"D:\Leaf_classifier\data\valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
valid_generator.reset()
valid_pred = model.predict(valid_generator, steps=valid_generator.samples // valid_generator.batch_size + 1)
valid_labels_true = valid_generator.classes[:len(valid_pred) * 32][:valid_generator.samples]
valid_acc = np.mean(np.argmax(valid_pred, axis=1) == valid_labels_true)
print(f"Re-evaluated Validation Accuracy: {valid_acc:.4f}")