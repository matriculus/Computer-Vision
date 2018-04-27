import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self):
		# flag for new line
		self.new_line = True
		# was the line detected in the last iteration?
		self.non_detect_tally = 0
		self.non_detect_tally_max = 5
		# self.detected = False  
		# x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None
		
		self.ploty = []
		
		# camera parameters
		self.M = 0
		self.Minv = 0
		
		# windows
		# Set the width of the windows +/- margin
		self.margin = 50

		# Set minimum number of pixels found to recenter window
		self.minpix = 100
		
		# Conversions in x and y from pixels space to meters
		self.ym_per_pix = 30/720.0 # meters per pixel in y dimension
		self.xm_per_pix = 3.7/700.0 # meters per pixel in x dimension
	
	def perspective_transform(self, cols, rows):
		rows = 0.9*rows
		src_points = np.float32([[0.4*cols,0.6*rows],
								 [0*cols,rows],
								 [cols,rows],
								 [0.6*cols,0.6*rows]])
		dst_points = np.float32([[0*cols,0*rows],
								 [0*cols,rows],
								 [cols,rows],
								 [cols,0*rows]])
		# obtain perspective transform parameters
		self.M = cv2.getPerspectiveTransform(src_points, dst_points)
		self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)

	def thresh_bin_im(self, image):
		"""
		Return the colour thresholds binary for L, S and R channels in an image
		img: RGB image
		s_thresh: S channel threshold
		sxl_thresh: sobel x threshold for L channel
		sxr_thresh: sobel x threshold for R channel
		"""
		def bin_it(image, threshold):
			output_bin = np.zeros_like(image)
			output_bin[(image >= threshold[0]) & (image <= threshold[1])]=1
			return output_bin

		hls_im = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

		bin_thresh = [10, 255]
		# rgb threshold for yellow
		lower = np.array([140,110,0],dtype = "uint8")
		upper = np.array([255, 255, 170],dtype = "uint8")
		mask = cv2.inRange(image, lower, upper)
		rgb_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
		rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
		rgb_y = bin_it(rgb_y, bin_thresh)


		# rgb for white (best)
		lower = np.array([100,100,200],dtype = "uint8")
		upper = np.array([255, 255, 255],dtype = "uint8")
		mask = cv2.inRange(image, lower, upper)
		rgb_w = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
		rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
		rgb_w = bin_it(rgb_w, bin_thresh)


		# hls yellow
		lower = np.array([20,110,80],dtype = "uint8")
		upper = np.array([44,150,255],dtype = "uint8")
		mask = cv2.inRange(hls_im, lower, upper)
		hls_y = cv2.bitwise_and(hls_im, hls_im, mask = mask).astype(np.uint8)
	#     hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2BGR)
		hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
		hls_y = bin_it(hls_y, bin_thresh)

		im_bin = np.zeros_like(rgb_y).astype(np.uint8)
		im_bin [(hls_y == 1)|(rgb_w==1)|(rgb_y==1)]= 1
		
		return im_bin
	
	def bin_image(self, image):
		"""
		convert colour binary image into a single channel binary image
		image: array with 3 channels
		return binary image with 1 channel which is a bitwise or operation of the original colour channels
		"""
		binary = np.zeros_like(image[:,:,1])
		binary[(image[:,:,0] == 1) | (image[:,:,1]==1) |(image[:,:,2]==1)] = 1
		return binary
	
	def transform_n_warp(self, image):
		"""
		apply transforms to image and warp the image according to transformation matrix
		image: image to be transformed
		M: transformation matrix
		"""
		# image size
		img_size = image.shape[:2][::-1]
		self.perspective_transform(img_size[0], img_size[1])
		# warp image
		image = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)

		# convert to coloured binary image
		image = self.thresh_bin_im(image)

		return image

	def update_fit(self, left_fit, right_fit, patience=5):
		"""
		update the fit values for n iterations
		"""
		self.current_fit = [left_fit, right_fit]
		self.recent_xfitted.append(self.current_fit)
		if len(self.recent_xfitted) > patience:
			self.recent_xfitted.pop(0)
		
		# calculate best fit, mean of last n iterations determined by patience
		self.best_fit = np.mean(self.recent_xfitted, axis=0)
		
		return self.best_fit
		
	# find lane function, with sliding window approach
	def find_lane(self, warped):
		"""
		return image with lane lines using sliding window approach
		warped: image in bird's eye view
		"""
		# check if there is a previous line
#         if self.detected:
#             self.new_line = False

		# Set the width of the windows +/- margin
		margin = self.margin

		# Set minimum number of pixels found to recenter window
		minpix = self.minpix
		
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((warped, warped, warped)).astype(np.int8)#*255
		window_img = np.zeros_like(out_img)
		
		if self.new_line or self.non_detect_tally<self.non_detect_tally_max:
			# we first take a histogram along all the columns in the lower half of the image
			histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

			# Find the peak of the left and right halves of the histogram
			# These will be the starting point for the left and right lines
			midpoint = np.int(histogram.shape[0]/2)
			leftx_base = np.argmax(histogram[:midpoint])
			rightx_base = np.argmax(histogram[midpoint:]) + midpoint

			# Number of sliding windows
			nwindows = 9

			# Height of windows
			window_height = np.int(warped.shape[0]/nwindows)

			# Identify the x and y positions of all nonzero pixels in the image
			nonzero = warped.nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])

			# Current positions to be updated for each window
			leftx_current = leftx_base
			rightx_current = rightx_base

			# Create empty lists to receive left and right lane pixel indices
			left_lane_inds = []
			right_lane_inds = []

			# Step through the windows one by one
			for window in range(nwindows):
				# Identify window boundaries in x and y (and right and left)
				win_y_low = warped.shape[0] - (window+1)*window_height
				win_y_high = warped.shape[0] - window*window_height
				win_xleft_low = leftx_current - margin
				win_xleft_high = leftx_current + margin
				win_xright_low = rightx_current - margin
				win_xright_high = rightx_current + margin


#                 # Draw the windows on the visualization image (not necessary here)
#                 cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 5) 
#                 cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 5) 

				# Identify the nonzero pixels in x and y within the window
				good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
				good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]


				# Append these indices to the lists
				left_lane_inds.append(good_left_inds)
				right_lane_inds.append(good_right_inds)
				# If you found > minpix pixels, recenter next window on their mean position
				if len(good_left_inds) > minpix:
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				if len(good_right_inds) > minpix:        
					rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

			# Concatenate the arrays of indices
			left_lane_inds = np.concatenate(left_lane_inds)
			right_lane_inds = np.concatenate(right_lane_inds)
			
			self.new_line = False
			
		else:
			nonzero = warped.nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			
			left_fit, right_fit = self.best_fit
			
			left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
			right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 
		
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		# update global left and right fits
		left_fit, right_fit = self.update_fit(left_fit, right_fit)
		
		# Generate x and y values for plotting
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
		
		# save ploty
		self.ploty = ploty
		
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# change the colour of nonzero pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [100, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 100]
		
		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Define y-value where we want radius of curvature
		# We choose the maximum y-value, corresponding to the bottom of the image
		y_eval = np.max(ploty)
		
#         left_fit, right_fit = self.best_fit
#         left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#         right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(ploty*self.ym_per_pix, left_fitx*self.xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*self.ym_per_pix, right_fitx*self.xm_per_pix, 2)
		
		# Calculate the new radii of curvature
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		
		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

		# out_img = out_img.astype(np.int8)
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)        
		
		# %% Warp the detected lane boundaries back onto the original image
		# # Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(result, np.int_([pts]), (0,255, 0))
		

		return result, left_curverad, right_curverad
	
	def display_on_frame(self, image, left_curverad, right_curverad, car_off):
		"""
		Display texts on image using passed values
		"""
		# create display texts on image
		font = cv2.FONT_HERSHEY_COMPLEX
		curve_disp_txt = 'Curvature: Right = ' + str(np.round(right_curverad,2)) + 'm, Left = ' + str(np.round(left_curverad,2)) + 'm' 

		off_disp_txt = 'Car off by ' + str(np.round(car_off,2)) + 'm'

		cv2.putText(image, curve_disp_txt, (30, 60), font, 1, (0,0,0), 2)
		cv2.putText(image, off_disp_txt, (30, 90), font, 1, (0,0,0), 2)
		
		return image

	def pipeline(self, original_frame):
		# Changing the colour from BGR to RGB
		frame = original_frame.copy()
		""" 
		perform all advanced lane finding process on frame        
		and return processed frame
		"""
		# smooth the image using gaussian blur
		frame = cv2.medianBlur(frame,3)
		
		# undistort frame
		#frame = cv2.undistort(frame, self.mtx, self.dst, None, self.mtx)
		
		
		# transform image to bird's eye view
		frame = self.transform_n_warp(frame)
		
		# find lane lines 
		# Check that there is new line
		# Then check that new line detections makes sense, 
		# i.e., expected curvature, separation, and parallel.
		frame, left_curverad, right_curverad = self.find_lane(frame)
#         print(left_curverad)
#         print(right_curverad)
		
#         print abs(left_curverad - right_curverad)
		
		img_size = (frame.shape[:2][0], frame.shape[:2][1])
		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(frame.astype(np.float32), self.Minv, img_size[::-1]) 
		newwarp = np.uint8(newwarp)

		# Combine the result with the original image
		frame = cv2.addWeighted(original_frame, 1, newwarp, 0.6, 0)
		
		# calculate lane midpoint
		# left line intercept on x axis
		left_fit, right_fit = self.best_fit

		left_intcpt = left_fit[0]*img_size[1]**2 + left_fit[1]*img_size[1] + left_fit[2]

		# right line intercept on x axis
		right_intcpt = right_fit[0]*img_size[1]**2 + right_fit[1]*img_size[1] + right_fit[2]

		lane_mid = (left_intcpt + right_intcpt)/2.0

		car_off = lane_mid - img_size[0]/2.0

		# convert to meters
		car_off *= self.xm_per_pix

		# display visuals on frame
		frame = self.display_on_frame(frame, left_curverad=left_curverad, right_curverad=right_curverad,
						car_off=car_off)
		
		# return processed frame
		ret_frame = frame
		return ret_frame.astype(np.uint8)