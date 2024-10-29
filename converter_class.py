import os
import cv2
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import itertools
import json

class VideoToData:

    def __init__(self, path_to_video):

        self.path_to_video = path_to_video
        self.tensors = []
        self.annotations = []
        self.tensors_processed = []
        self.annotations_processed = []
        self.tensors_resized = []
        self.annotations_resized = []
        self.frames_set_import = []  # Fixed indentation
        self.annotations_import = []  # Fixed indentation

    def plot_boxes(self,image,annotation):

        # Convert tensor to numpy array for plotting
        # image_np = image.permute(1, 2, 0).numpy()
        image= (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label = annotation["class"]
        xmin = annotation["xmin"]
        ymin = annotation["ymin"]
        xmax = annotation["xmax"]
        ymax = annotation["ymax"]

        # Draw rectangle and label
        cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{label}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show the image with bounding boxes
        cv2.imshow("Test", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the window after displaying


    def get_frames(self, num_frames=100):
        cap = cv2.VideoCapture(self.path_to_video)  # Open video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
        interval = total_frames // num_frames  # Frames wanted by user
        count = 0
        extracted_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()  # Read video frame
            if not ret:
                break

            horizontal_flip = transforms.RandomHorizontalFlip(p=1)
            vertical_flip = transforms.RandomVerticalFlip(p=1) 

            if count % interval == 0 and extracted_frames < 4*num_frames:  # Check every wanted frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

                tensor = F.to_tensor(frame_rgb)  # Transform to tensor
                self.tensors.append(tensor)  # Append to final list
                extracted_frames += 1

                flipped_tensor_horizontally = horizontal_flip(tensor.unsqueeze(0)).squeeze(0)  # Flip and remove the batch dimension
                self.tensors.append(flipped_tensor_horizontally)  # Append flipped tensor
                extracted_frames += 1

                flipped_tensor_vertically = vertical_flip(tensor.unsqueeze(0)).squeeze(0)  # Flip and remove the batch dimension
                self.tensors.append(flipped_tensor_vertically)  # Append flipped tensor
                extracted_frames += 1

                flipped_tensor = horizontal_flip(tensor.unsqueeze(0))  # Add batch dimension
                flipped_tensor = vertical_flip(flipped_tensor).squeeze(0)  # Flip vertically and remove batch dimension
                self.tensors.append(flipped_tensor)  # Append flipped tensor

                print(f"Extracted frame {extracted_frames}: Shape {tensor.shape}")

            count += 1

        cap.release()

    def process_frame(self, frame, object_position, object_size, tag, corner, filename):
        x_center, y_center = object_position  # Center of the object
        width, height = object_size  # Size of the bounding box

        # Convert frame (PIL image) to NumPy array for indexing
        frame_np = np.array(frame)

        print("---debug : frame_np : {frame_np.shape}")

        frame_np = np.transpose(frame_np, (1, 2, 0))

        # Get original image dimensions
        original_height, original_width = frame_np.shape[0], frame_np.shape[1]

        print("---debug : frame height and wighth  : {original_height} and {original_width}")

        # Define crop dimensions (2/3 of the original width and height)
        crop_width = int(original_width * (2 / 3))
        crop_height = int(original_height * (2 / 3))

        # Initialize crop coordinates based on the corner
        if corner == "top-left":
            crop_x1, crop_y1 = 0, 0
        elif corner == "top-right":
            crop_x1, crop_y1 = original_width - crop_width, 0
        elif corner == "bottom-left":
            crop_x1, crop_y1 = 0, original_height - crop_height
        elif corner == "bottom-right":
            crop_x1, crop_y1 = original_width - crop_width, original_height - crop_height
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

    def process_set(self, object_position, object_size, tag, corner, name):
        for i, frame in enumerate(self.tensors):
            filename = name + "_" + str(i)  # Create a unique filename
            tensor, annotation = self.process_frame(frame, object_position, object_size, tag, corner, filename)
            self.annotations_processed.append(annotation)
            self.tensors_processed.append(tensor)
            print(f"Processed frame {i}: {filename} with annotation {annotation}")

    def fix_resize(self, target_size=(224, 224)):
        for i, image in enumerate(self.tensors_processed):
            if image.size == 0:
                print(f"Warning: Image at index {i} is empty.")
                continue
            annotation_new = self.annotations_processed[i]
            original_height, original_width = image.shape[:2]
            target_width, target_height = target_size

            resized_image = cv2.resize(image, target_size)
            self.tensors_resized.append(resized_image)

            scale_x = target_width / original_width
            scale_y = target_height / original_height

            # Adjust and clamp bounding box coordinates
            x_min = max(0, min(int(annotation_new['xmin'] * scale_x), target_width))
            y_min = max(0, min(int(annotation_new['ymin'] * scale_y), target_height))
            x_max = max(0, min(int(annotation_new['xmax'] * scale_x), target_width))
            y_max = max(0, min(int(annotation_new['ymax'] * scale_y), target_height))

            # Save resized annotation
            self.annotations_resized.append({
                'filename': annotation_new['filename'],
                'xmin': x_min,
                'ymin': y_min,
                'xmax': x_max,
                'ymax': y_max,
                'class': annotation_new['class']
            })

            print(f"Processed frame {i}: with annotation {self.annotations_resized[-1]}")

        for i in range(4):
            index = np.random.randint(1,len(self.tensors_resized))
            self.plot_boxes(self.tensors_resized[index],self.annotations_resized[index])

    def import_video_shuffle(self, num_frames, object_position, object_size, tag, target_size):
        self.get_frames(num_frames)  # Fixed self.

        corners = ["top-left", "top-right", "bottom-left", "bottom-right"]

        for corner in corners:
            self.process_set(object_position, object_size, tag, corner, corner)  # Fixed self.

        print(f'Length of tensor processed: {len(self.tensors_processed)}')
        print(f'Lenght of annotations processed: {len(self.annotations_processed)}')
        

        self.fix_resize(target_size)  # Call the method directly with target_size

    def save_image_and_metadata(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(len(self.tensors_resized)):
            pic = self.tensors_resized[i]
            if pic.dtype != np.uint8:
                pic = np.clip(pic * 255, 0, 255).astype(np.uint8)
            
            # Check if the image is in the correct shape (height, width, 3)
            if len(pic.shape) == 3 and pic.shape[0] == 3:
                pic = np.transpose(pic, (1, 2, 0))

            image_path = os.path.join(folder_path, self.annotations_resized[i]["filename"] + '.jpg')
            success = cv2.imwrite(image_path, pic)

            if not success:
                print(f"Failed to save image {self.annotations_resized[i]['filename']}.jpg")
            else:
                print(f"Saved image: {image_path}")
        
        metadata_path = os.path.join(folder_path, 'metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(self.annotations_resized, f, indent=4)



