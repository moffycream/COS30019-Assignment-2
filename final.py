import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime
from sklearn.metrics import classification_report # Use scikit-learn to calculate classification report
import cv2

def CNN():
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

    saved_model_path = "./cnn"
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

def photo_predict_CNN():
    # Load the saved model
    cnn_model_path = "./cnn"
    cnn_model = tf.saved_model.load(cnn_model_path)

    # Choose a photo using a file dialog
    photo_path = filedialog.askopenfilename(initialdir="./testing", title="Select a Photo", filetypes=[("Image Files", "*.jpg;*.jpeg")])

    if photo_path:
        # Load the selected photo
        image = cv2.imread(photo_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (224, 224))  # Resize the image to the model's input size

        # Display the selected photo
        plt.imshow(image)
        plt.title("Selected Photo")
        plt.axis('off')
        plt.show()

        # Preprocess the image
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Apply the model to the selected photo
        # Use the correct input signature
        prediction_scores = cnn_model(tf.constant(image, dtype=tf.float32), training=False)
        predicted_index = np.argmax(prediction_scores)

        # Map class index to class name
        class_names = ['happy', 'sad']  # Replace with your actual class names

        predicted_class = class_names[predicted_index]

        # Print the prediction
        print(f"Predicted class: {predicted_class}")


def run_selected_model():
    selected_model = model_var.get()
    print(f"Running {selected_model} model")

    # Here you can call the corresponding function to run the selected model
    if selected_model == 'CNN':
        photo_predict_CNN()
    elif selected_model == 'SVM':
        run_svm_model()

def run_cnn_model():
    # Add your code to run the CNN model here
    CNN()

def run_svm_model():
    # Add your code to run the SVM model here
    print("SVM function")

def train_selected_model():
    selected_model = model_var.get()
    print(f"Training {selected_model} model")

    if selected_model == 'CNN':
        CNN()
    elif selected_model == 'SVM':
        # train_svm_model()  # Implement train_svm_model function
        print("SVM function")

def start_camera_feed():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        return

    photo_count = 1
    testing_folder = "testing"

    # Create the testing folder if it doesn't exist
    if not os.path.exists(testing_folder):
        os.makedirs(testing_folder)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to capture and save a photo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Use the current date and time to create a unique filename
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            photo_path = os.path.join(testing_folder, f"photo_{current_datetime}_{photo_count}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"Photo saved: {photo_path}")
            photo_count += 1

        # Break the loop if 'esc' key is pressed
        elif cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Create the main Tkinter window
root = tk.Tk()
root.title("Model Selection")

# Set the window size (width x height)
window_width = 400
window_height = 300
root.geometry(f"{window_width}x{window_height}")

# Create a label
label = ttk.Label(root, text="Select a model to run:")
label.pack(pady=10)

# Create a Combobox (dropdown) for model selection
models = ['CNN', 'SVM']
model_var = tk.StringVar()
model_combobox = ttk.Combobox(root, textvariable=model_var, values=models, state="readonly")
model_combobox.pack(pady=10)
model_combobox.set(models[0])  # Set the default selection

# Create a button to train the selected model
train_button = ttk.Button(root, text="Train Model", command=train_selected_model)
train_button.pack(pady=10)

# Create a button to run the selected model
run_button = ttk.Button(root, text="Run Selected Model", command=run_selected_model)
run_button.pack(pady=10)

# Create a button to start the camera feed
camera_button = ttk.Button(root, text="Start Camera Feed (press Q to take photo, ESC to exit)", command=start_camera_feed)
camera_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()