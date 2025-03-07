import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Enable GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected")

# Data directories
train_dir = r"D:\Leaf classifier\data\train"
valid_dir = r"D:\Leaf classifier\data\valid"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Check class distribution
train_labels = [f.split(os.sep)[-2] for f in train_generator.filepaths]
class_counts = Counter(train_labels)
print("Class distribution:", class_counts)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Register the FixedDropout layer
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

# Load the existing model
MODEL_PATH = r"D:\Leaf classifier\improvedd_leaf_classifier.keras"
base_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'FixedDropout': FixedDropout}
)

# Freeze all layers except the last few
for layer in base_model.layers[:-10]:  # Unfreeze the last 10 layers
    layer.trainable = False

# Update the final layer for the new number of classes
x = base_model.layers[-2].output  # Get the second-to-last layer
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # New output layer

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=1e-5)  # Use a lower learning rate for fine-tuning
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=20,  # Fewer epochs for fine-tuning
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Save the updated model
model.save(r"D:\Leaf classifier\improvedd_leaf_classifier_with_grapes.keras")
print("Model saved successfully!")