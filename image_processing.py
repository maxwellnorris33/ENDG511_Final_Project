import logging
import cv2
import mediapipe as mp
import os
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk
import numpy as np

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
        self.panel.grid(padx=10, pady=10)

        # Destructor function to release resources
        parent.protocol('WM_DELETE_WINDOW', self.destructor)

        # Start the video loop
        self.video_loop()

        # Bind the 's' key to the save_current_image method to save images on key press
        parent.bind('<s>', self.save_current_image)

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
                x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
                x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
                y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
                y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h

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
            pil_image = Image.fromarray(cv2image)

            # Detect hand and process the image
            result = self.hand_detection_function(cv2image)
            if result:
                x_min, y_min, x_max, y_max = result
                hand_region = pil_image.crop((x_min, y_min, x_max, y_max))

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
            pil_image_resized = pil_image.resize((28, 28), Image.LANCZOS)
            pil_image_bw = pil_image_resized.convert('L')
            pil_image_normalized = np.asarray(pil_image_bw) / 255.0
            pil_image_final = Image.fromarray(np.uint8(pil_image_normalized * 255))  # Convert back to PIL image for display

            # Convert the image for Tkinter and display it
            imgtk = ImageTk.PhotoImage(image=pil_image_final)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

        # Schedule the next frame read
        self.after(30, self.video_loop)

    def save_current_image(self, event=None):
        if self.current_hand_region is not None:
            # Convert the NumPy array back to a PIL Image object
            image_to_save = Image.fromarray(np.uint8(self.current_hand_region * 255))

            # Save the PIL Image
            cropped_image_path = os.path.join(self.output_path, f"hand_crop_{self.image_counter}.png")
            image_to_save.save(cropped_image_path)
            logging.info(f"Saved cropped hand region to {cropped_image_path}")
            self.image_counter += 1
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
