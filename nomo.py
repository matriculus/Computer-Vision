# import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from collections import deque
import pyautogui
queue_length=5

class Vehicle:
	def straight():
		pyautogui.keyDown('w')
		pyautogui.keyUp('s')
		pyautogui.keyUp('a')
		pyautogui.keyUp('d')

	def left():
		pyautogui.keyDown('w')
		pyautogui.keyUp('s')
		pyautogui.keyDown('a')
		pyautogui.keyUp('d')

	def right():
		pyautogui.keyDown('w')
		pyautogui.keyUp('s')
		pyautogui.keyUp('a')
		pyautogui.keyDown('d')

	def slow():
		pyautogui.keyUp('w')
		pyautogui.keyUp('s')
		pyautogui.keyUp('a')
		pyautogui.keyUp('d')

class LaneDetector:
	def __init__(self):
		self.left_lines = deque(maxlen=queue_length)
		self.right_lines = deque(maxlen=queue_length)
	
	def process(self, image):
		rows, cols = image.shape[:2]
		white_yellow = select_white_yellow(image)
		gray = convert_gray_scale(white_yellow)
		smooth_gray = apply_smoothing(gray)
		edges = detect_edges(smooth_gray)
		regions = select_region(edges)
		lines = hough_lines(regions)
		print([None if lines==None else len(lines)])
		try:
			left_line, right_line = lane_lines(image, lines, rows)
			
			def mean_line(line, lines):
				if line is not None:
					lines.append(line)
				
				if len(lines)>0:
					line = np.mean(lines, axis=0, dtype=np.int32)
					line = tuple(map(tuple, line))
				return line

			left_line = mean_line(left_line, self.left_lines)
			right_line = mean_line(right_line, self.right_lines)
			
			return draw_lane_lines(image, (left_line, right_line))
		except:
			return image


# def show_images(images, cmap=None):
# 	cols = 2
# 	rows = (len(images)+1)/cols
# 	plt.figure(figsize=(10,11))
# 	for i, image in enumerate(images):
# 		plt.subplot(rows,cols,i+1)
# 		cmap = 'gray' if len(image.shape)==2 else cmap
# 		plt.imshow(image, cmap=cmap)
# 		plt.xticks([])
# 		plt.yticks([])
# 	plt.tight_layout(pad=0, h_pad=0, w_pad=0)
# 	plt.show()

def select_white_yellow(image):
	converted = convert_hls(image)
	# white masking, choosing high Light value, did not filter Hue and Saturation
	lower = np.uint8([100, 100, 200])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(image, lower, upper)
	# yellow masking, choosing Hue and high Saturation
	lower = np.uint8([20, 120, 80])
	upper = np.uint8([45, 200, 255])
	yellow_mask = cv2.inRange(converted, lower, upper)
	# combining masks
	mask = cv2.bitwise_or(white_mask, yellow_mask)
	masked = cv2.bitwise_and(image, image, mask=mask)
	return masked

def perspective_transform(img, warping="warp"):
	rows, cols = img.shape[:2]
	src = np.float32([[0, 0.7*rows],
					[0.3*cols, 1/2*rows],
					[0.7*cols, 1/2*rows],
					[cols, 0.7*rows]])

	dst = np.float32([[0, rows],
					[0, 0],
					[cols, 0],
					[cols, rows]])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	if warping=="unwarp":
		return cv2.warpPerspective(img, Minv, (cols, rows))
	else:
		return cv2.warpPerspective(img, M, (cols, rows))

def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(img, mask)
	return masked

def histogram(image):
	return np.sum(image[int(image.shape[0]/2):,:], axis=0)

def draw_lines(img, lines):
	try:
		#print('Lines detected: '+str(lines.shape[0]))
		for line in lines:
			coord = line[0];
			x1,y1,x2,y2 = coord
			cv2.line(img, (x1,y1), (x2,y2), [255,255,255], 2)
	except:
		pass

def average_slope_intercept(lines):
	left_lines=[]
	left_weights=[]
	right_lines=[]
	right_weights=[]
	for line in lines:
		for x1,y1,x2,y2 in line:
			if x2==x1:
				continue
			slope = (y2-y1)/(x2-x1)
			intercept = y1-slope*x1
			length = np.sqrt((y2-y1)**2+(x2-x1)**2)
			if slope<0:
				left_lines.append((slope, intercept))
				left_weights.append((length))
			else:
				right_lines.append((slope, intercept))
				right_weights.append((length))
	# add more weight to long lines
	left_lane = np.dot(left_weights, left_lines)/np.sum(left_weights) if len(left_weights) > 0 else None
	right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights) > 0 else None
	return left_lane, right_lane

#print(list_of_lines)

def make_line_points(y1,y2,line):
	if line is None:
		return None
	slope, intercept = line
	if 0<= slope <= 0.05:
		slope = 0.05
	elif -0.05 <= slope <= 0:
		slope = -0.05
	x1 = int((y1-intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	
	return ((x1, y1), (x2, y2))

def lane_lines(image, lines, rows):
	left_lane, right_lane = average_slope_intercept(lines)
	#print(left_lane[0], right_lane[0])
	y1 = 0.9*rows
	y2 = y1*0.6
	left_line = make_line_points(y1, y2, left_lane)
	right_line = make_line_points(y1, y2, right_lane)
	return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=5):
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def convert_hls(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2HLS) # convert from rgb to hls

def convert_gray_scale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=7):
	# kernel size must be positive and odd
	return cv2.GaussianBlur(image, (kernel_size,kernel_size),0)

def detect_edges(image, low_threshold=40, high_threshold=160):
	return cv2.Canny(image, low_threshold, high_threshold)

def select_region(image):
	# makes other regions than selected as 0
	rows, cols = image.shape[:2]
	vertices = np.array([[0.1*cols, rows],
							[0.4*cols, 0.6*rows],
							[0.6*cols, 0.6*rows],
							[0.9*cols, rows]], dtype="int32")
	return roi(image, [vertices])

def hough_lines(image):
	return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def combined_edges(image):
	HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# HLS_laplacian = cv2.Laplacian(HLS,cv2.CV_64F)
	# HSV_laplacian = cv2.Laplacian(HSV,cv2.CV_64F)
	# laplacian = cv2.Laplacian(image, cv2.CV_64F)
	HLS_Canny = cv2.Canny(HLS, 50, 100)
	HSV_Canny = cv2.Canny(HSV, 50, 100)
	Canny = cv2.Canny(image, 50, 100)
	CT_edge = cv2.addWeighted(HLS_Canny, 0.5, HSV_Canny, 0.5, 0)
	result = cv2.addWeighted(CT_edge, 0.8, Canny, 0.2, 0)
	# result = CT_edge
	return result