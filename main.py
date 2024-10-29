import os
import tkinter as tk
from tkinter import filedialog, messagebox
from converter_class import VideoToData
from model_handler_class import TrainModel
import sys
import io
import threading  # Import threading module
from analyze_class import Analyze_video

# Redirect print to console
class PrintLogger(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end of the console

# DEFINITION

def transform_video():
    threading.Thread(target=run_transform_video).start()  # Run the function in a new thread

def run_transform_video():
    # This is where the long-running task is performed
    video_path = filedialog.askopenfilename()
    export_path = filedialog.askdirectory()
    if video_path and export_path:  # Check if paths are valid
        video = VideoToData(video_path)
        video.import_video_shuffle(10,(540, 540), (360, 360), "cat", (224, 224))
        video.save_image_and_metadata(export_path)
        print("Video exported!")
    else:
        print("Video or export path not selected.")

def train_model():
    threading.Thread(target=run_train_model).start()  # Run in a new thread

def run_train_model():
    current_folder = filedialog.askdirectory()
    json_path = filedialog.askopenfilename(filetypes=[("Json files",".json")])
    model_ = TrainModel(current_folder, json_path)
    model_.load_image_and_transform()
    model_.create_model()
    model_.train_model()
    model_.save_model()
    print("Model trained and saved!")

def load_model():
    threading.Thread(target=run_load_model).start()  # Run in a new thread

def run_load_model():
    current_folder = filedialog.askdirectory()
    json_path = filedialog.askopenfilename()
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth;*.h5")])
    if model_path:  # Check if a model path was selected
        model_ = TrainModel(current_folder, json_path)
        model_.load_image_and_transform()  # Fixing the model initialization
        model_.load_model(model_path, "train")
        model_.train_model()
        model_.save_model()
        print("Model loaded!")

def run_analyze_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    model_path = filedialog.askopenfilename(filetypes=[("Model", "*.pth")])
    frames = Analyze_video(video_path)
    frames.open_video()
    frames.test_random_frame(model_path, threeshold = 0.1)

def analyze_video():
    threading.Thread(target=run_analyze_video).start()

def log_message(message):
    console.insert(tk.END, message + "\n")
    console.see(tk.END)  # Scroll to the end of the console


# APP

app = tk.Tk()
app.title("Software")
app.geometry("400x500")
app.configure(bg='black')

# Button styling
button_style = {
    'width': 15,
    'height': 2,
    'fg': 'white',
    'font': ('Arial', 10),
    'borderwidth': 1,
    'relief': 'solid'
}

# Function to create labeled buttons
def create_labeled_button(app, label_text, bg_color, button_text, command):
    label = tk.Label(app, text=label_text, bg='black', fg='white', font=('Arial', 10))
    label.pack(pady=(5, 0))  # Add some padding
    button = tk.Button(app, text=button_text, bg=bg_color, command=command, **button_style)
    button.pack(pady=5)

# Create buttons with labels
create_labeled_button(app, "Transform video in to frames and data augmentation","#4affb0", "Transform Video", transform_video)
create_labeled_button(app, "Create model and train it", "#0089ab", "Train Model", train_model)
create_labeled_button(app, "Load model and train it","#006e4f","Load Model", load_model)
create_labeled_button(app,"Analyze video","#a87000","Analyze Video", analyze_video)
# Redirecting print to console
console = tk.Text(app, height=10, width=50, bg='dimgray', fg='white', font=('Courier New', 10))  # Slightly lighter background
console.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
sys.stdout = PrintLogger(console)

# Label to display the folder path
folder_label = tk.Label(app, text="", bg='black', fg='blue', font=('Arial', 10))
folder_label.pack(pady=5)

app.mainloop()


