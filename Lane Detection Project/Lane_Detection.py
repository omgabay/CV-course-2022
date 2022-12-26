# -*- coding: utf-8 -*-
"""Lane_Detection_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13S3XmvfvQsSgZJKgl685wCwbFnytnulg
"""

import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt
figsize = (30, 30)
debug_mode = True

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image) 

    x1_bottom_right= 970
    y1_bottom_right= height-10

    x2_bottom_left=  200
    y2_bottom_left= height-10

    x3_upper_right= 620
    y3_upper_right= 500

    x4_upper_left= 520
    y4_upper_left= 500
    
    pts = np.array([[(x4_upper_left,y4_upper_left), (x2_bottom_left, y2_bottom_left), (x1_bottom_right, y1_bottom_right),(x3_upper_right, y3_upper_right)]] , np.int32)
    cv.fillPoly(mask, pts, 255)  
    masked_image = cv.bitwise_and(image,mask)
    return masked_image

def apply_color_mask(gray_img,color_img):
  # Converting the image to hsv color space 
  img_hsv = cv.cvtColor(color_img,cv.COLOR_RGB2HSV)

  lower_yellow = np.array([20,100,100], dtype = "uint8")
  upper_yellow = np.array([30,255,255], dtype = "uint8")
  mask_yellow = cv.inRange(img_hsv, lower_yellow,upper_yellow)
 
  lower_white = np.array([0, 0, 231])
  upper_white = np.array([180, 18, 255])
  mask_white  = cv.inRange(gray_img, 170, 255)
  #mask_white = cv.inRange(img_hsv, lower_white, upper_white)
  
  mask_yw = cv.bitwise_or(mask_white, mask_yellow)
  final_img = cv.bitwise_and(gray_img, mask_yw)
  return final_img

def canny(img):  
    kernel = 7
    blurred = cv.GaussianBlur(img,(kernel, kernel),0)
    canny = cv.Canny(blurred, 150, 200, None, 3) 
    return canny

def display_lines(image, lines,color=(0,0,255)): 
  lines_img = np.copy(image)
  line_stroke = 2
    
  for line in lines:
    if line is not None:
      x1, y1, x2, y2 = line.reshape(4)
      cv.line(lines_img , (x1,y1) , (x2,y2), color, line_stroke)         

  return lines_img

def add_text_to_image(text,image, position=(385,100)): 
  font = cv.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  color = (255,255,255) 
  thickness = 2
  image = cv.putText(image, text,position, font, fontScale, color, thickness, cv.LINE_AA) 
  return image

text = ""
def draw_lane_polygon(image, line1, line2,present,frames):  
  lane_image = np.zeros_like(image)
  
  if line1 is None or line2 is None:
    return lane_image, frames

  x1, y1, x2, y2 = line1.reshape(4)
  x3,y3,x4,y4 = line2.reshape(4)
  
  max_base = max(abs(x1 - x2), abs(x3-x4))  
  points = np.array([[(x1,y1), (x2, y2),(x4, y4),(x3,y3)]] , np.int32)
  cv.fillPoly(lane_image, points, color=(230, 0, 0))
  if max_base <= 240 or frames > 0:
    #update number of frames to print message about lane-change   
    if frames == 0: 
      global text
      text = ""      
      frames = 85      
      if not present[0]:
        text = "Car moved to the left lane"
      elif not present[1]:
        text = "Car moved to the right lane"   
    add_text_to_image(text,lane_image)    
    frames -= 1
  base_text = f'base length {max_base}px'  
  add_text_to_image(base_text,lane_image, position=(430,150))   #DEBUG MODE 
  return lane_image, frames

def make_coordinates(image , line_parameters):
  slope , intercept = line_parameters
  y1 = int(image.shape[0])
  y2 = int(0.75*y1)
  x1 = int((y1-intercept) / slope)
  x2 = int((y2-intercept)/ slope)
  return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines,prev_l,prev_r):
  left_fit    = []
  right_fit   = [] 
  present= [True,True]
  averaged_lines = [] 
  lines_filtered = []
     
  if lines is not None:    
    for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      slope , intercept = np.polyfit((x2,x1), (y2,y1), 1)   

      if abs(slope) < 0.5:   # 0.4 before
        continue      
      lines_filtered.append(line) #FOR TEST    

      # left lane lines have negative slope
      if slope < 0:
        left_fit.append((slope, intercept))    
      # right lane lines have positive slope 
      if slope > 0:
        right_fit.append((slope,intercept))      

  if len(left_fit) > 0:
    left_fit_average = np.average(left_fit, axis=0)
    slope,intercept = left_fit_average
    left_line =  make_coordinates(image, left_fit_average)
    text_l = f'left line slope={slope} x_val={left_line[0]}'
    # if debug_mode:
    #   print(text_l)       
  else:
    left_line = prev_l
    present[0] = False 

  if len(right_fit)> 0:
    right_fit_average = np.average(right_fit, axis=0)
    slope,intercept = right_fit_average
    right_line = make_coordinates(image, right_fit_average)
    text_r = f'right line slope={slope} x_val={right_line[0]}' 
    # if debug_mode:
    #   print(text_r) 
  else:
    right_line = prev_r
    present[1] = False  
  
  averaged_lines = [left_line, right_line] 
  
  return present,averaged_lines, lines_filtered

"""# Lane Detection On Video

"""

def main():    
    cap = cv.VideoCapture("test_final.mp4")
    video_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  
    video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  
    video_fps = int(cap.get(cv.CAP_PROP_FPS))
    size = (video_width, video_height)
    print(f'reading video (fps={video_fps})')
    print(f'video dimension ({video_width,video_height})')

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%H_%M")

    # Below VideoWriter object will create video stored in 'output.avi' file.
    video = cv.VideoWriter(f'output_{dt_string}.avi', 
                            cv.VideoWriter_fourcc(*'MJPG'),
                            video_fps, size)
    if debug_mode:                         
        test_video = cv.VideoWriter(f'rays_roi_view_{dt_string}.avi', cv.VideoWriter_fourcc(*'MJPG'),video_fps, size)                        
                            

    prev_l = None
    prev_r = None
    frames = 0   
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or cv.waitKey(1) & 0xFF == ord('q'):
            break 
        
        frame = frame.astype(np.uint8)
        frame_out = np.copy(frame)    
        canny_img = canny(frame_out)   
        canny_masked_img = region_of_interest(canny_img)    
        
        lines = cv.HoughLinesP(canny_masked_img, 2, np.pi/180, 100,np.array([]),minLineLength=10,maxLineGap=70)  
        
        present,averaged_lines, filtered_lines = average_slope_intercept(frame, lines,prev_l,prev_r)
        prev_l,prev_r = averaged_lines[0],averaged_lines[1]
    
        polygon_img , frames =  draw_lane_polygon(frame,averaged_lines[0],averaged_lines[1],present,frames)
        combo_img = cv.addWeighted(frame, 0.85, polygon_img, 1, 1) 

        # for testing purposes -------------------------------------------------------------------- 
        if debug_mode:    
            masked_frame = region_of_interest(frame_out)      
            lines_to_display = averaged_lines
            test_img = display_lines(masked_frame, lines_to_display,color=(0,0,255))
            cv.imshow("lines in roi", test_img)
            test_video.write(test_img)
        # ------------------------------------------------------------------------------------------
        
        #Writing video and display image 
        cv.imshow("result", combo_img)
        video.write(combo_img)  
    
    cap.release()
    video.release()
    test_video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()