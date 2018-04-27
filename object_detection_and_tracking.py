# coding: utf-8
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)

# Reference Code:
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict, deque
from io import StringIO
from PIL import ImageGrab
import cv2
# from nomo import LaneDetector, Vehicle
from ALD import Line

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("models/research/")
sys.path.append("models/research/slim")

# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
object_detection_path = "models/research/object_detection"
PATH_TO_LABELS = os.path.join(object_detection_path,'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# print("retrieved")
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
# 	file_name = os.path.basename(file.name)
# 	if 'frozen_inference_graph.pb' in file_name:
# 		tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

LD = Line()

def box2pixels(box, rows, cols):
	ymin = int(box[0]*rows)
	xmin = int(box[1]*cols)
	ymax = int(box[2]*rows)
	xmax = int(box[3]*cols)
	b = (xmin, ymin, xmax-xmin, ymax-ymin)
	return b

buffer_frame = 15

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		n = 1
		while True:
			timer = cv2.getTickCount()
			# print(n)
			image_np = cv2.resize(np.array(ImageGrab.grab(bbox=(0,45, 640,525)).convert('RGB')), (320, 240))
			image_t = image_np
			rows, cols = image_t.shape[:2]
			try:
				new_screen = LD.pipeline(image_np)
			except:
				new_screen = image_np
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			if n == 1:
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=5,
					max_boxes_to_draw=20)

				# print(scores[0]>0.9)
				n_boxes = len(scores[scores>0.5])
				tracker = []
				pts = []
				for i in range(n_boxes):
					bbox = box2pixels(boxes[0][i], rows, cols)
					p1 = (int(bbox[0]), int(bbox[1]))
					tracker.append(cv2.TrackerKCF_create())
					tracker[i].init(image_t, bbox)
					pts.append(deque(maxlen=buffer_frame))
					pts[i].appendleft((int(bbox[0]), int(bbox[1])))

			# ok = True
			for i in range(n_boxes):
				ok, bbox = tracker[i].update(image_t)

				if ok:
					p1 = (int(bbox[0]), int(bbox[1]))
					p2 = (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1]))
					pts[i].appendleft(p1)
					pts_len = len(pts[i])
					cv2.rectangle(image_t, p1, p2, (255,0,0), 2, 1)
					[cv2.line(image_t, pts[i][j-1], pts[i][j], (0,0,255), 2) for j in range(1, pts_len)]
				else:
					cv2.putText(image_t, "Tracking failure detected", (100, 80),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			cv2.putText(image_t, "FPS: {}".format(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (50, 170, 50), 2)

			image_np = cv2.addWeighted(image_np, 0.6, new_screen, 0.4, 0.0)
			image_np = cv2.addWeighted(image_np, 0.6, image_t, 0.4, 0.0)
			if n == buffer_frame: n = 1
			else: n += 1
			cv2.imshow("window",cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
			cv2.moveWindow("window", 1000, 50)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break