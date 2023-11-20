import os
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.client import device_lib

print("Available devices:", device_lib.list_local_devices())

print("TF version:", tf.version)
print("Hub version:", hub.version)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

model_name = "mobilenet_v3_large_075_224"
model_handle_map = {
    "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

# Get the current working directory where your Python file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path to your dataset directory
data_dir = os.path.join(current_dir, 'train')

model_handle = model_handle_map.get(model_name)
pixels = 224  # Adjust the image size for the specific model

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 16

# Load the training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Load the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Define the class names
class_names = train_ds.class_names

# Build and train the model
do_fine_tuning = False

print("Building model with", model_handle)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(model_handle, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# Plot training history
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()

# Save the trained model
saved_model_path = "./saved_models"
tf.saved_model.save(model, saved_model_path)

# Visualize images during testing
for x_batch, y_batch in val_ds.take(1):  # Take one batch from the validation dataset
    for i in range(min(5, BATCH_SIZE)):  # Display up to 5 images
        image = x_batch[i].numpy().astype("uint8")
        true_label = class_names[np.argmax(y_batch[i])]
        predicted_scores = model.predict(np.expand_dims(x_batch[i], axis=0))
        predicted_label = class_names[np.argmax(predicted_scores)]
        
        plt.figure()
        plt.imshow(image)
        plt.title(f'True label: {true_label}, Predicted label: {predicted_label}')
        plt.axis('off')
        plt.show()
