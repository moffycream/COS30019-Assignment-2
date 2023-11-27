import tkinter as tk
from tkinter import ttk # Import ttk module for combobox
from tkinter import filedialog # Import filedialog module for opening files
import cv2 # Import OpenCV module for camera feed
import os # Import os module for creating folders
from datetime import datetime

import model_1 as model_1 # Import model_1.py
import model_2 as model_2 # Import model_2.py

# Ask the user to select a photo
def select_photo():
    print("Selecting photo...")

    # Ask the user to select a photo
    photo_path = filedialog.askopenfilename(initialdir="./testing", title="Select a Photo", filetypes=[("Image Files", "*.jpg;*.jpeg")])

    # Check if the user has selected a photo
    if photo_path:
        print("Photo selected: ", photo_path)
    else:
        print("No photo selected")

    return photo_path

# Train the selected model
def train_model():
    # Get the selected model
    model = model_var.get()

    # Check if the selected model is CNN 1
    if model == models[0]:
        model_1.train()
    # Check if the selected model is CNN 2
    elif model == models[1]:
        model_2.train()

# Run the selected model
def run_model():

    # Get the selected model
    model = model_var.get()

    # Check if the selected model is CNN 1
    if model == models[0]:
        print("Running model 1...")
        
        # Ask the user to select a photo
        photo_path = select_photo()

        # Check if the user has selected a photo
        if photo_path:
            # Predict the photo
            model_1.predict(photo_path)


    # Check if the selected model is CNN 2
    elif model == models[1]:
        print("Running model 2...")

        # Ask the user to select a photo
        photo_path = select_photo()

        # Check if the user has selected a photo
        if photo_path:
            # Predict the photo
            model_2.predict(photo_path)

# Start the camera feed
def start_camera_feed():
    print("Starting camera feed...")

    # Initialize the camera
    cap = cv2.VideoCapture(0) # 0 = default camera

    # Check if the camera is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Create a testing folder if it doesn't exist
    if not os.path.exists("testing"):
        os.makedirs("testing")
    
    # Loop until the user presses ESC
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Camera feed', frame)

        # Wait for the user to press ESC
        c = cv2.waitKey(1)
        if c == 27:
            break

        # Wait for the user to press Q to take a photo
        if c == ord('q'):
            # Use the current date and time as the filename
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = "testing/" + now + ".jpg"
            cv2.imwrite(filename, frame)
            print("Photo saved: ", filename)
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    

# Tkinter interface
root = tk.Tk()
root.title("Model Selection")

# Set the window size (width x height)
root.geometry("400x300")

# Create a label widget
label = tk.Label(root, text="Select a model to run:")
label.pack(pady=10)

# Create a Combobox (dropdown) for model selection
models = ['CNN 1', 'CNN 2']
model_var = tk.StringVar()
model_combobox = ttk.Combobox(root, textvariable=model_var, values=models, state="readonly")
model_combobox.pack(pady=10)
model_combobox.set(models[0])  # Set the default selection

# Create a button to train the model
train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(pady=10)

# Create a button to run the model
run_button = tk.Button(root, text="Run Selected Model", command=run_model)
run_button.pack(pady=10)

# Create a button to start the camera feed
camera_button = tk.Button(root, text="Start Camera Feed (press Q to take photo, ESC to exit)", command=start_camera_feed)
camera_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()