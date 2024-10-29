from model_handler_class import TrainModel
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch
from converter_class import VideoToData


class Analyze_video:

	def __init__(self,path):
		self.video_path = path
		self.video_frames = []

	def open_video(self):
		self.video_frames = cv2.VideoCapture(self.video_path)
		print(f"Frame imported ({int(self.video_frames.get(cv2.CAP_PROP_FRAME_COUNT))})")

	def draw_boxes(self,image, boxes, labels, scores, threshold=0.5):
		for box, label, score in zip(boxes,labels,scores):
			if score >= threshold:
				xmin, ymin, xmax, ymax = map(int,box)
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0),2)
				cv2.putText(image, f"{label}: {score:.2f}", (xmin, ymin - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		return image

	def test_random_frame(self, model_path, threeshold):
		model_ = TrainModel(None,None)
		model = model_.load_model(model_path,"eval")
		
		total_frames = int(self.video_frames.get(cv2.CAP_PROP_FRAME_COUNT))

		random_frame = np.random.randint(0,total_frames-1)
		self.video_frames.set(cv2.CAP_PROP_POS_FRAMES,random_frame)

		ret, frame = self.video_frames.read()
		if not ret:
			print("Failed to retrieve frame")
			return

		image_tensor = F.to_tensor(frame)

		with torch.no_grad():
			predictions = model([image_tensor])

		boxes = predictions[0]["boxes"].cpu().numpy()
		scores = predictions[0]["scores"].cpu().numpy()
		print(scores)
		labels = ["cat" if label == 1 else "background" for label in predictions[0]["labels"].cpu().numpy()]

		frame_with_boxes = self.draw_boxes(frame, boxes, labels, scores, threeshold)

		cv2.imshow("Random Frame with Detections", frame_with_boxes)
		cv2.waitKey(0)  # Wait for a key press to close the window
		cv2.destroyAllWindows()

		self.video_frames.release()

	def video_analyze(self,threeshold):
		return True
