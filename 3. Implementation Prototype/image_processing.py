import logging
import cv2
import mediapipe as mp
import os
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

class ASL_vision_gui(ttk.Frame):
    def __init__(self, parent, output_path="./Fingers"):
        super().__init__(parent)
        self.grid()

        # Set up output directory for saving images
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Initialize counter for image filenames and variable for current hand region
        self.image_counter = 0
        self.current_hand_region = None

        # Initialize video capture for the default camera
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            logging.error("Failed to open default camera. Exiting...")
            exit()
        
        # Label for displaying the image
        self.panel = ttk.Label(self)
        self.panel.grid(padx=1, pady=1)
        
        self.pil_font = ImageFont.truetype("arial.ttf", 30)

        # Destructor function to release resources
        parent.protocol('WM_DELETE_WINDOW', self.destructor)

        #load in model
        self.model = tf.keras.models.load_model(f'trained_signLang_model_copy.h5')
        self.model.compile(optimizer='rmsprop', # I chose this because adam would not work
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
        
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Start the video loop
        self.video_loop()

        
    @staticmethod
    def hand_detection_function(frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the bounding box for the hand
                h, w, _ = frame.shape
                x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w *1/1.2
                x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w  *1.2
                y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h *1/1.2
                y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h *1.2

                # Adjust the bounding box
                buffer = 30
                x_min, y_min = max(x_min - buffer, 0), max(y_min - buffer, 0)
                x_max, y_max = min(x_max + buffer, w), min(y_max + buffer, h)

                return x_min, y_min, x_max, y_max
        return None

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            # Convert to RGB and create a PIL image
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.pil_image = Image.fromarray(cv2image)

            # Detect hand and process the image
            result = self.hand_detection_function(cv2image)
            if result:
                x_min, y_min, x_max, y_max = result
                hand_region = self.pil_image.crop((x_min, y_min, x_max, y_max))

                # Resize the cropped hand region to 28x28 pixels
                hand_region_resized = hand_region.resize((28, 28), Image.LANCZOS)

                # Convert the image to grayscale ('L' mode for 8-bit pixels, black and white)
                hand_region_bw = hand_region_resized.convert('L')

                # Convert the PIL image to a NumPy array for normalization
                hand_region_array = np.asarray(hand_region_bw)

                # Normalize the pixel values by dividing by 255.0
                self.current_hand_region = hand_region_array / 255.0
            else:
                self.current_hand_region = None

            # Resize, convert to grayscale, and normalize the full image for display
            pil_image_resized = self.pil_image.resize((28, 28), Image.LANCZOS)
            pil_image_bw = pil_image_resized.convert('L')
            pil_image_normalized = np.asarray(pil_image_bw) / 255.0
            pil_image_final = Image.fromarray(np.uint8(pil_image_normalized * 255))  # Convert back to PIL image for display
            
            #evaluate the model
            self.evaluate_current_image()
            
            # Convert the image for Tkinter and display it
            imgtk = ImageTk.PhotoImage(image=self.pil_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            
        # Schedule the next frame read
        self.after(30, self.video_loop)

    def evaluate_current_image(self, event=None):
        if self.current_hand_region is not None:

            self.imgArray = pd.DataFrame(self.current_hand_region)
            self.y_pred = self.model.predict(self.imgArray.values.reshape(-1, 28, 28, 1))
            self.predicted_class = self.labels[np.argmax(self.y_pred, axis = 1)[0]]

            #add text
            draw = ImageDraw.Draw(self.pil_image)
            draw.text((20, 20), self.predicted_class, font=self.pil_font, fill='aqua')

        else:
            logging.info("No hand region to save")

    def destructor(self):
        logging.info("Closing application...")
        self.vs.release()
        cv2.destroyAllWindows()
        self.master.destroy()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    root.title("ASL Vision GUI")
    app = ASL_vision_gui(root)
    root.mainloop()
