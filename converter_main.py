import os
import cv2
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import itertools
import json


def get_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)  # open video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total num of frames
    interval = total_frames // num_frames  # num_frames wanted by user
    tensors = []
    count = 0
    extracted_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()  # open vid
        if not ret:
            break

        if count % interval == 0 and extracted_frames < num_frames:  # check every wanted frames
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # get frames
            tensor = F.to_tensor(frame_rgb)  # transform to tensor
            tensors.append(tensor)  # append to final list
            extracted_frames += 1
            print(f"Extracted frame {extracted_frames}: Shape {tensor.shape}")

        count += 1

    cap.release()
    return tensors

def process_frame(frame, object_position, object_size, tag, corner, filename):
    x_center, y_center = object_position  # Center of the object
    width, height = object_size  # Size of the bounding box

    # Convert frame (PIL image) to NumPy array for indexing
    frame_np = np.array(frame)
    frame_np = np.transpose(frame_np, (1, 2, 0))

    # Get original image dimensions
    original_height, original_width = frame_np.shape[0], frame_np.shape[1]

    # Define crop dimensions (2/3 of the original width and height)
    crop_width = int(original_width * (2 / 3))
    crop_height = int(original_height * (2 / 3))

    # Initialize crop coordinates based on the corner
    if corner == "top-left":
        crop_x1 = 0
        crop_y1 = 0
    elif corner == "top-right":
        crop_x1 = original_width - crop_width
        crop_y1 = 0
    elif corner == "bottom-left":
        crop_x1 = 0
        crop_y1 = original_height - crop_height
    elif corner == "bottom-right":
        crop_x1 = original_width - crop_width
        crop_y1 = original_height - crop_height
    else:
        raise ValueError("Invalid corner specified. Choose from ['top-left', 'top-right', 'bottom-left', 'bottom-right'].")

    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    # Ensure cropping coordinates stay within the image bounds
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(original_width, crop_x2)
    crop_y2 = min(original_height, crop_y2)

    # Crop the image
    cropped_image = frame_np[crop_y1:crop_y2, crop_x1:crop_x2]

    # Calculate new bounding box coordinates in the cropped image
    new_xmin = max(0, int((x_center - width / 2) - crop_x1))
    new_ymin = max(0, int((y_center - height / 2) - crop_y1))
    new_xmax = min(crop_width, int((x_center + width / 2) - crop_x1))
    new_ymax = min(crop_height, int((y_center + height / 2) - crop_y1))

    # Create metadata for bounding box
    metadata = {
        'filename': filename,  # Use the provided filename
        'xmin': new_xmin,
        'ymin': new_ymin,
        'xmax': new_xmax,
        'ymax': new_ymax,
        'class': tag
    }

    return cropped_image, metadata

def process_set(set_frame, object_position, object_size, tag, corner, name):
    annotations = []
    tensors = []

    for i, frame in enumerate(set_frame):
        filename = name + "_" + str(i)  # Fixed: cast i to string for concatenation
        tensor, annotation = process_frame(frame, object_position, object_size, tag, corner, filename)
        annotations.append(annotation)
        tensors.append(tensor)
        print(f'Processed {i} frames for corner: {corner}')

    return tensors, annotations

def fix_resize(zoomed_images, annotations, target_size=(224, 224)):
    tensors_new = []
    annotations_new = []

    for i, image in enumerate(zoomed_images):
        # Check if the image is empty
        if image.size == 0:
            print(f"Warning: Image at index {i} is empty.")
            print(image)
            continue  # Skip processing this image
        
        annotation = annotations[i]
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size

        # Resize the image to target dimensions using OpenCV
        resized_image = cv2.resize(image, target_size)
        tensors_new.append(resized_image)

        # Calculate scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Adjust and clamp bounding box coordinates
        x_min = max(0, min(int(annotation['xmin'] * scale_x), target_width))
        y_min = max(0, min(int(annotation['ymin'] * scale_y), target_height))
        x_max = max(0, min(int(annotation['xmax'] * scale_x), target_width))
        y_max = max(0, min(int(annotation['ymax'] * scale_y), target_height))

        # Save resized annotation
        annotations_new.append({
            'filename': annotation['filename'],
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max,
            'class': annotation['class']
        })

    return tensors_new, annotations_new

def import_video_shuffle(video_path, num_frames, object_position, object_size, tag, target_size):
    frames = get_frames(video_path, num_frames)
    frames_set = []
    annotations = []

    corner = ["top-left", "top-right", "bottom-left", "bottom-right"]

    for i in range(len(corner)):
        frame_set, annotation = process_set(frames, object_position, object_size, tag, corner[i], corner[i])
        frames_set.append(frame_set)
        annotations.append(annotation)

    flattened_list = list(itertools.chain.from_iterable(frames_set))
    flattened_dict_list = [d for sublist in annotations for d in sublist]

    resized_list, resized_annotation = fix_resize(flattened_list,flattened_dict_list,target_size)

    return resized_list, resized_annotation

def save_image_and_metadata(image, metadata, folder_path):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    for i in range(len(image)):
        # Save the image
         # Ensure the image is in uint8 format
        pic = image[i]
        if pic.dtype != np.uint8:
            pic = np.clip(image[i] * 255, 0, 255).astype(np.uint8)
        
        # Check if the image is in the correct shape (height, width, 3)
        if len(pic.shape) == 3 and pic.shape[0] == 3:
            # If the channel is the first dimension, transpose it to (height, width, 3)
            pic = np.transpose(pic, (1, 2, 0))

        image_path = os.path.join(folder_path, metadata[i]["filename"] + '.jpg')
        success = cv2.imwrite(image_path, pic)

        if not success:
            print(f"Failed to save image {metadata[i]['filename']}.jpg")
        else:
            print(f"Saved image: {image_path}")
        
        
    metadata_path = os.path.join(folder_path, 'metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


video_path = "C:\\Users\\nelso\\Downloads\\IMG_6987.mov"

object_position = [540, 540]
object_size = [360, 360]

a = import_video_shuffle(video_path, 50, object_position, object_size, "cat", (224, 224))

save_image_and_metadata(a[0],a[1],"C:\\Users\\nelso\\Desktop\\Software\\generated_images")
