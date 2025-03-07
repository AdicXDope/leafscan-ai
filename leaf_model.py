import os
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
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

# Debug: Check if paths exist
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Valid directory exists: {os.path.exists(valid_dir)}")

# Debug: List files in the train directory
if os.path.exists(train_dir):
    print("Train directory contents:", os.listdir(train_dir))
else:
    print("Train directory not found.")

# Debug: List files in the valid directory
if os.path.exists(valid_dir):
    print("Valid directory contents:", os.listdir(valid_dir))
else:
    print("Valid directory not found.")

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

# Load EfficientNet-B3 with Fine-Tuning
base_model = efn.EfficientNetB3(weights='imagenet', include_top=False)

# Freeze base layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Unfreeze top layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Combine the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=1e-4)
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
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights_dict
)

# Save the model
model.save(r"D:\Leaf classifier\improvedd_leaf_classifier.keras")
print("Model saved successfully!")