import os
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.models.detection as detection
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import cv2
import json
import time
import numpy as np


class TrainModel:

    def __init__(self, images_path, metadata_path):
        self.images_path = images_path
        self.metadata_path = metadata_path
        self.model = None
        self.image = None
        self.annotations = []
        self.tensor_pics = []

    def plot_boxes(self,annotation):

        # Convert tensor to numpy array for plotting
        image_np = self.image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

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

    def load_image_and_transform(self):
        """Load images and transform them"""

        images = [
            os.path.join(self.images_path, filename)
            for filename in os.listdir(self.images_path)
            if filename.endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"{len(images)} images loaded.")

        # Define the transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)
            print("Metadata loaded")
            print(f'Metadata len : {len(metadata)}')
            print(f'Images len : {len(images)}')

        for image in images:
            pic = Image.open(image).convert('RGB')
            transformed_pic = transform(pic)
            self.tensor_pics.append(transformed_pic)

            filename = os.path.basename(image).split(".")[0]
            image_metadata = next((item for item in metadata if item.get("filename") == filename), None)

            if image_metadata:
                self.annotations.append(image_metadata)
        
        # Randomly select images to plot boxes
        for k in range(1,4):  # Ensure we don't exceed available images
            index = np.random.randint(0, len(self.tensor_pics))
            self.image = self.tensor_pics[index]
            annotation = self.annotations[index]
            self.plot_boxes(annotation)  # Call the plot_boxes method

    def create_model(self):
        """Create the object detection model"""
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
        num_classes = 2  # Assuming background + one class
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        print("Model created")

    def save_model(self, path="model.pth"):
        """Save the model state to the specified path"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, mode):
        """Load model weights from the specified path"""
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.model.load_state_dict(torch.load(path))
        if mode == "eval":
            self.model.eval()  # Set to evaluation mode
        elif mode == "train":
            self.model.train()  # Set to training mode

        return self.model

    def train_model(self):
        """Train the object detection model"""
        self.model.train()
        
        # Define the optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        num_epochs = 5

        # Register hooks to track intermediate activations
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
                print(f"Intermediate activation - {name}: {output.shape}")
            return hook
        
        for name, layer in self.model.named_children():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                layer.register_forward_hook(get_activation(name))

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            total_loss = 0

            for i, row in enumerate(self.annotations):
                batch_start_time = time.time()

                # Load image and target annotations
                image_tensor = self.tensor_pics[i]
                target = {
                    "boxes": torch.tensor([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64)  # Label for 'cat'
                }

                optimizer.zero_grad()
                outputs = self.model([image_tensor], [target])
                
                # Calculate and accumulate the loss
                loss = sum(loss for loss in outputs.values())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Log detailed batch info
                print(f"Batch [{i + 1}/{len(self.annotations)}], Batch Loss: {loss.item():.4f}, Time: {time.time() - batch_start_time:.2f}s")

            # Calculate average loss for the epoch and log
            avg_loss = total_loss / len(self.annotations)
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - epoch_start_time:.2f}s, Average Loss: {avg_loss:.4f}")

