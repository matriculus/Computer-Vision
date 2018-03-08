import numpy as np
from PIL import ImageGrab
import cv2
#import pyscreenshot as ImageGrab
import time
print("All libraries imported!")

def process_img(image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=50, threshold2=100)
    return processed_img
    

def main():
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,60, 640,480)))
        new_screen = process_img(screen)
        last_time = time.time()
        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
