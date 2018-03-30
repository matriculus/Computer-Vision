import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_balance(gray_image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray_image)
	return gray

def bin_image(image):
	"""
	convert colour binary image into a single channel binary image
	image: array with 3 channels
	return binary image with 1 channel which is a bitwise or operation of the original colour channels
	"""
	binary = np.zeros_like(image[:,:,1])
	binary[(image[:,:,0] == 1) | (image[:,:,1]==1) |(image[:,:,2]==1)] = 1
	return binary

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def thresh_bin_im(img):
	"""
	Return the colour thresholds binary for L, S and R channels in an image
	img: RGB image
	"""    
	s_thresh = [170, 255]
	sx_thresh = [25, 200]
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	r_channel = img[:,:,0]
	
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	
	# sobel x on r channel
	sobelx_r = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0)
	abs_sobelx_r = np.absolute(sobelx_r)
	scaled_sxr = np.uint8(255*abs_sobelx_r/np.max(abs_sobelx_r))

	sxrbinary = np.zeros_like(scaled_sxr)
	sxrbinary[(scaled_sxr >= sx_thresh[0]) & (scaled_sxr <= sx_thresh[1])] = 1
	
	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	empty = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	# Stack each channel
	# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# be beneficial to replace this channel with something else.
	color_binary = np.dstack((sxbinary, empty, empty))

	return color_binary

def bin_it(image, threshold):
	"""
	converts a single channeled image to a binary image,
	using upper and lower threshold
	"""
	assert len(image.shape) == 2, 'Image received has more than one channel'
	
	output_bin = np.zeros_like(image)
	output_bin[(image >= threshold[0]) & (image <= threshold[1])]=1
	return output_bin

def threshold_colours(image):
	"""
	Return binary image from thresholding colour channels
	img: RGB image
	"""
	# convert image to hls colour space
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
	
	# binary threshold values
	bin_thresh = [20, 255]
	
	# rgb thresholding for yellow
	lower = np.array([140,110,0],dtype = "uint8")
	upper = np.array([255, 255, 170],dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	rgb_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
	rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
	rgb_y = bin_it(rgb_y, bin_thresh)
	
	
	# rgb thresholding for white (best)
	lower = np.array([100,100,200],dtype = "uint8")
	upper = np.array([255, 255, 255],dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	rgb_w = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
	rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
	rgb_w = bin_it(rgb_w, bin_thresh)
	
	
	# hls thresholding for yellow
	lower = np.array([20,110,80],dtype = "uint8")
	upper = np.array([44, 150, 255],dtype = "uint8")
	mask = cv2.inRange(hls, lower, upper)
	hls_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
	hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)
	hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
	hls_y = bin_it(hls_y, bin_thresh)
	
	im_bin = np.zeros_like(hls_y)
	im_bin [(rgb_y == 1)|(rgb_w==1)|(hls_y==1)]= 1
	
	return im_bin*255

def perspective_transform(cols, rows):
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
	M = cv2.getPerspectiveTransform(src_points, dst_points)
	Minv = cv2.getPerspectiveTransform(dst_points, src_points)
	return M, Minv

def transform_n_warp(image):
	# applying transformations to image
	# image size
	img_size = image.shape[:2][::-1]
	# convert to coloured binary
	image = threshold_colours(image)
	# warping image
	M, _ = perspective_transform(img_size[0], img_size[1])
	image = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
	return image


img1 = cv2.imread("06.jpg")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
warped = transform_n_warp(img)
lower_region = warped[int(warped.shape[0]/2):, :]
histogram = np.sum(lower_region, axis=0)

out_img = np.dstack((warped, warped, warped))

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

# Set the width of the windows +/- margin
margin = 100

# Set minimum number of pixels found to recenter window
minpix = 50

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

	# Draw the windows on the visualization image
	cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 5) 
	cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 5) 

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


# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
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

# Draw the lane onto the warped blank image
window_img = np.zeros_like(out_img)
cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

# out_img = out_img.astype(np.int8)
result = cv2.addWeighted(out_img, 1, window_img, 0.25, 0)

y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720.0 # meters per pixel in y dimension
xm_per_pix = 3.7/700.0 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print('Left:',left_curverad, 'm | Right:', right_curverad, 'm')

# %% Warp the detected lane boundaries back onto the original image
# # Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(result, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
img_size = img.shape[:2][::-1]
_, Minv = perspective_transform(img_size[0], img_size[1])
newwarp = cv2.warpPerspective(result, Minv, (img_size[0], img_size[1])) 

newwarp = np.uint8(newwarp)
print(img.shape, newwarp.shape)
# Combine the result with the original image
final = cv2.addWeighted(img1, 1, newwarp, 0.5, 0)

# %% Output visual display of the lane boundaries and numerical estimation of the lane curvature and vehicle position


# calculate lane midpoint
# left line intercept on x axis
left_intcpt = left_fit[0]*img_size[1]**2 + left_fit[1]*img_size[1] + left_fit[2]

# right line intercept on x axis
right_intcpt = right_fit[0]*img_size[1]**2 + right_fit[1]*img_size[1] + right_fit[2]

lane_mid = (left_intcpt + right_intcpt)/2.0

car_off = (lane_mid - img_size[0]/2.0)*xm_per_pix

def display_on_frame(image, left_curverad, right_curverad, car_off):
	"""
	Display texts on image using passed values
	"""
	# create display texts on image
	font = cv2.FONT_HERSHEY_COMPLEX
	curve_disp_txt = 'Curvature: Right = ' + str(np.round(right_curverad,2)) + 'm, Left = ' + str(np.round(left_curverad,2)) + 'm' 

	off_disp_txt = 'Car off by ' + str(np.round(car_off,2)) + 'm'

	cv2.putText(final, curve_disp_txt, (30, 60), font, 1, (0,0,0), 2)
	cv2.putText(final, off_disp_txt, (30, 90), font, 1, (0,0,0), 2)
	
	return image

final = display_on_frame(final, left_curverad=left_curverad, right_curverad=right_curverad,
						car_off=car_off)

plt.plot(histogram)
plt.show()

cv2.imshow("Frame", final)
cv2.waitKey(0)
cv2.destroyAllWindows()