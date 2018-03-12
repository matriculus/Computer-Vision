import numpy as np
import time
import keyboard
import cv2
import os
from PIL import ImageGrab

file_name = "training_data.npy"

if os.path.isfile(file_name):
	print("File exists, loading previous data!")
	training_data = list(np.load(file_name))
else:
	print("Trained file does not exist, creating a new one!")
	training_data = []

def key_pressed():
	output = [0, 0, 0, 0, 0] # [a,w,s,d, none]
	if keyboard.is_pressed('a'):
		print('a is pressed')
		output[0] = 1
	elif keyboard.is_pressed('s'):
		print('s is pressed')
		output[2] = 1
	elif keyboard.is_pressed('d'):
		print('d is pressed')
		output[3] = 1
	elif: keyboard.is_pressed('w'):
		print('w is pressed when none')
		output[1] = 1
	else:
		print('No key pressed')
		output[-1] = 1

	return output

def main():
	last_time = time.time()
	while(True):
		screen = np.array(ImageGrab.grab(bbox=(0,45, 640,525)))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (80,60))
		output = key_pressed()
		training_data.append([screen, output])
		print(screen.shape)
		print(output)
		print(time.time()-last_time)
		last_time = time.time()
		if len(training_data) % 10 == 0:
			print(len(training_data))
			print('Saving Data...')
			np.save(file_name, training_data)

		cv2.imshow('screen', screen)
		if cv2.waitKey(25) == ord('p'):
			cv2.destroyAllWindows()
			break

main()