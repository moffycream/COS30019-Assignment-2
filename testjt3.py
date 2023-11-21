import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pathlib
from sklearn.metrics import classification_report # Use scikit-learn to calculate classification report

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

model_name = "mobilenet_v3_large_075_224"
model_handle_map = {
    "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

model_handle = model_handle_map.get(model_name)
pixels = 224  # Adjust the image size for the specific model

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 8

# Set the data directory to the 'train' directory in the same location as the script
data_dir = os.path.join(os.path.dirname(__file__), 'train')

# Check if the directory exists
if not os.path.exists(data_dir):
    print(f"The specified local path '{data_dir}' does not exist.")
else:
    print(f"Local data directory: {data_dir}")

def get_class_names(data_dir):
    return sorted(item.name for item in pathlib.Path(data_dir).glob("*/") if item.is_dir())

class_names = get_class_names(data_dir)

def build_dataset(subset, validation_split=0.2):
    if subset == "testing":
        return tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="validation",  # Use "validation" subset for testing
            label_mode="categorical",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE)
    else:
        return tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset=subset,
            label_mode="categorical",
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE)

# Split the data into training (80%), validation (10%), and test (10%)
train_ds = build_dataset("training", validation_split=0.1).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = build_dataset("validation", validation_split=0.1).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = build_dataset("testing", validation_split=0.1).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Calculate the cardinality of each set
train_size = tf.data.experimental.cardinality(train_ds).numpy()
val_size = tf.data.experimental.cardinality(val_ds).numpy()
test_size = tf.data.experimental.cardinality(test_ds).numpy()

# Create class_names tuple after any modifications to train_ds
class_names = get_class_names(data_dir)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])
do_data_augmentation = False
if do_data_augmentation:
    preprocessing_model.add(tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(tf.keras.layers.RandomTranslation(0.2, 0))
    preprocessing_model.add(tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(tf.keras.layers.RandomFlip(mode="horizontal"))
train_ds = train_ds.map(lambda images, labels: (preprocessing_model(images), labels))

val_ds = build_dataset("validation")
valid_size = val_ds.cardinality().numpy()
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels: (normalization_layer(images), labels))

do_fine_tuning = False

print("Building model with", model_handle)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(model_handle, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

# Set steps_per_epoch and validation_steps
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = tf.data.experimental.cardinality(val_ds).numpy() // BATCH_SIZE

hist = model.fit(
    train_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps
).history

plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.legend()

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.legend()

saved_model_path = "./saved_models"
tf.saved_model.save(model, saved_model_path)

# Visualize all images along with true and predicted labels from the test set
true_labels = []
predicted_labels = []

for x, y in test_ds:
    for i in range(len(x)):
        image = x[i, :, :, :]
        true_index = np.argmax(y[i])

        # Apply the same normalization to the test image as done during training
        image = normalization_layer(image)

        # Expand the test image to (1, 224, 224, 3) before predicting the label
        prediction_scores = model.predict(np.expand_dims(image, axis=0))
        predicted_index = np.argmax(prediction_scores)

        plt.figure(figsize=(12, 3))  # Adjust figure size as needed
        plt.imshow(image)
        plt.title(f'True: {class_names[true_index]}\nPredicted: {class_names[predicted_index]}')
        plt.axis('off')

        true_labels.append(true_index)
        predicted_labels.append(predicted_index)

plt.show()

# Calculate True Negative (TN), True Positive (TP), False Negative (FN), False Positive (FP)
conf_matrix = tf.math.confusion_matrix(true_labels, predicted_labels, num_classes=len(class_names))

# Extract values from the confusion matrix
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1] if conf_matrix.shape[1] > 1 else 0
FN = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 else 0
TP = conf_matrix[1, 1] if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 1 else 0

# Print the confusion matrix and values
print("Confusion Matrix:")
print(conf_matrix.numpy())
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Positive (TP): {TP}")

# Map class indices to class names for scikit-learn
class_names_scikit = [class_names[i] for i in range(len(class_names))]

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=class_names_scikit)

# Print the classification report
print("Classification Report:")
print(report)