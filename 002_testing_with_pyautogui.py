import numpy as np
from PIL import ImageGrab
import cv2
#import pyscreenshot as ImageGrab # Use it for linux
import time
import pyautogui
import nomo
import matplotlib.pyplot as plt
print("All libraries imported!")

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

def start_timer(n):
    for i in list(range(n))[::-1]:
        print(i+1)
        time.sleep(1)

def main():
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,45, 640,525)))
        new_screen, slopes, histogram = nomo.process_image(screen, perspective=False)
        print(slopes)
        
        cv2.imshow('window', new_screen)
        # plt.plot(histogram)
        # plt.show()
        turn = abs(slopes[0])/abs(slopes[1])-1

        if turn < -0.5:
            print("Right--->")
            #right()
        elif turn > 0.5:
            print("<---Left")
            #left()
        else:
            print("^Straight^")
            #straight()
        
        print('Look took {} seconds.'.format(time.time()-last_time))
        last_time = time.time()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

start_timer(3)
main()

