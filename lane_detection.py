import numpy as np
import cv2 
import glob, os
import matplotlib.pyplot as plt
import nomo
from nomo import LaneDetector

detector = LaneDetector()
test_images = [plt.imread(path) for path in glob.glob("gta_images/*.jpg")]
nomo.show_images(test_images)

final_image = list(map(detector.process, test_images))
nomo.show_images(final_image)

print(test_images[0].shape)