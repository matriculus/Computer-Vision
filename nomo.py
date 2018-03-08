import numpy as np
import cv2

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
    y1 = rows
    y2 = y1*0.4
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return (left_line, right_line), [left_lane[0], right_lane[0]]

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        if line is not None:
            cv2.line(image, *line, color, thickness)

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

def process_image(image, perspective=False):
    if perspective:
        processed_img = perspective_transform(image, warping="warp")
    else:
        processed_img = image
    processed_img = combined_edges(processed_img)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    rows, cols = processed_img.shape[:2]
    if perspective:
        vertices = np.array([[0,0],
                            [0, rows],
                            [0.4*cols, rows],
                            [0.4*cols, 0.1*rows],
                            [0.8*cols, 0.1*rows],
                            [0.8*cols, rows],
                            [cols,rows],
                            [cols, 0]], dtype="int")
    else:
        vertices = np.array([[0, 0.7*rows],
                            [0.4*cols, 0.4*rows],
                            [0.6*cols, 0.4*rows],
                            [cols, 0.7*rows],
                            [0.8*cols, 0.7*rows],
                            [3/5*cols, 1/2*rows],
                            [2/5*cols, 1/2*rows],
                            [0.3*cols, 0.7*rows]], dtype="int")
    processed_img = roi(processed_img, [vertices])
    histo = histogram(processed_img)
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 50, 5)
    slopes = [1,1]
    draw_lines(processed_img, lines)
    # try:
    #     lane_lines1, slopes = lane_lines(processed_img, lines, rows)
    #     draw_lane_lines(processed_img, lane_lines1)
    #     #draw_lines(processed_img, lines)
    # except:
    #     pass
    if perspective:
        processed_img = perspective_transform(processed_img, warping="unwarp")
    else:
        pass
    return processed_img, slopes, histo