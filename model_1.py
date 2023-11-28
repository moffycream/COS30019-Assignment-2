import tensorflow as tf
import tensorflow_hub as hub
import os
import pathlib # For handling file paths
import matplotlib.pyplot as plt # For plotting
import numpy as np
import sklearn.metrics as tnf # For calculating the confusion matrix and classification report

def plot_learning_curves(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
def visualize_images(test_data, y_pred, class_names):
    print("Visualizing images...")

    # Get the true labels
    y_true = test_data.classes

    # Total number of batches in the test set
    num_batches = len(test_data)

    # Loop through all batches
    for batch_index in range(num_batches):
        # Access the batch
        batch_images, batch_labels = test_data[batch_index]

        # Loop through all images within the batch
        for image_index in range(test_data.batch_size):
            # Check if the image index is within the actual number of images
            if batch_index * test_data.batch_size + image_index >= test_data.n:
                break

            # Display the image
            plt.figure(figsize=(4, 4))
            plt.imshow(batch_images[image_index])

            # Get the true and predicted labels
            true_label = class_names[y_true[batch_index * test_data.batch_size + image_index]]
            pred_label = class_names[np.argmax(y_pred[batch_index * test_data.batch_size + image_index])]

            # Get the predicted probabilities
            pred_probabilities = y_pred[batch_index * test_data.batch_size + image_index]

            # Display the true and predicted labels with percentages
            title = f"True: {true_label} | Predicted: {pred_label} ({pred_probabilities[1] * 100:.2f}% {class_names[1]}, {pred_probabilities[0] * 100:.2f}% {class_names[0]})"
            plt.title(title)

            # Show accuracy to predicted label
            plt.axis('off')
            plt.show()

def report(y_true, y_pred, class_names):
    # Calculate the confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred)

    # Extract values from the confusion matrix
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1] if conf_matrix.shape[1] > 1 else 0
    FN = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 else 0
    TP = conf_matrix[1, 1] if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 1 else 0

    # Draw the confusion matrix
    print("Confusion Matrix")
    print(conf_matrix.numpy())
    print(f"True Negative (TN): {TN}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print(f"True Positive (TP): {TP}")

    # Print the classification report
    print("Classification Report")
    report = tnf.classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
def predict(photo_path):
    print("Predicting...")

    # Load the saved model
    model_path = os.path.join(os.path.dirname(__file__), 'models/model_1')

    # Check if the model exists
    if not os.path.exists(model_path):
        print("Model not found")
        return

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load the image
    img = tf.keras.preprocessing.image.load_img(photo_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Normalize the image
    img_array = img_array / 255.0

    # Get the predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get the class names
    class_names = sorted(item.name.split('_')[0] for item in pathlib.Path(os.path.join(os.path.dirname(__file__), 'train')).glob("*/") if item.is_dir())

    # Show the image and predicted class name and probability
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Predicted class: {class_names[predicted_class]} ({predictions[0][predicted_class] * 100:.2f}%)")
    plt.axis('off')
    plt.show()

def train():
    print("Training...")

    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Check TensorFlow version
    print("TF Version: ", tf.__version__)
    print("Hub Version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
    
    # Load the MobileNet tf.keras model.
    classifier_model = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5"
    print("Loading model from: ", classifier_model)

    # Adjust the image size according to the model specification.
    pixels = 224
    IMAGE_SIZE = (pixels, pixels)
    BATCH_SIZE = 8

    # Set seeds for reproducibility
    seed_value = 123
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Load pre-trained model as Keras layer
    model = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SIZE+(3,)) # 3 is the number of channels in the image (RGB)
    ])

    # Freeze the pre-trained model weights
    for layer in model.layers:
        layer.trainable = False

    # Add a classification head
    model.add(tf.keras.layers.Dense(2, activation='softmax')) # 2 is the number of classes
    
    # Print the model summary
    model.summary()

    # Set the data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'train')

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print("Data directory not found")
        return
    
    # Check if the data directory contains the required subdirectories
    class_name = sorted(item.name.split('_')[0] for item in pathlib.Path(data_dir).glob("*/") if item.is_dir())

    if not class_name:
        print("Class directories not found")
        return
    
    #  Create a dataset
    # rescale=1/255 normalizes the image pixel values to be between 0 and 1
    # validation_split=0.2 reserves 20% of the images for validation set
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        validation_split=0.2,
    )

    def build_dataset(subset):
        return image_generator.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset=subset,
            shuffle=True,
            seed = seed_value,
        )
    
    train_data = build_dataset("training")
    validation_data = build_dataset("validation")

    # Define a separate testing set
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test')
    test_data = image_generator.flow_from_directory(
        test_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed = seed_value
    )
    test_data = test_data

    # Print the number of images in each subset
    print("Total number of images:", len(list(image_generator.flow_from_directory(data_dir).filenames)))
    print("Number of training images:", train_data.n)
    print("Number of validation images:", validation_data.n)
    print("Number of testing images:", test_data.n)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), # Adam optimizer with default parameters
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # Categorical crossentropy loss
        metrics=['accuracy'] # Accuracy metric
    )

    # Train the model
    epochs = 8
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data
    )

    # Save the model
    model.save('models/model_1', save_format='tf')


    # ==================== EVALUATION ====================

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(validation_data)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Plot the learning curves
    plot_learning_curves(history)

    # Get the true labels and predicted probabilities
    y_true = test_data.classes
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Ask the user to visualize the images
    visualize = input("Visualize images? (y/n): ")
    if visualize.lower() == 'y':
        # Visualize all the images along with true and predicted labels from the test set
        visualize_images(test_data, y_pred_probs, class_name)

    # Calculate the confusion matrix and classification report
    report(y_true, y_pred, class_name)
