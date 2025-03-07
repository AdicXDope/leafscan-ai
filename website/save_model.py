import tensorflow as tf
import efficientnet.tfkeras as efn

# Step 1: Create the model
base_model = efn.EfficientNetB3(weights='imagenet', include_top=False)

# Step 2: Add custom layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(4, activation='softmax')(x)

# Step 3: Combine the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Step 4: Save the model
model.save(r"D:\Leaf classifier\website\improved_leaf_classifier.keras")
print("Model saved successfully!")