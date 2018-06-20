#! /usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt


def tmplMatch(img_ins,img_ref,ref_rect,width,height,search_scale=1.6,show=False):
  """return res,ins_rect"""
  methods = ('cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED')
  low_thresh = 30
  high_thresh = low_thresh * 4.0

  temp = np.asmatrix([])
  edged_template = np.asmatrix([])
  template = np.asmatrix([])
  ins_region = np.asmatrix([])
  edged = np.asmatrix([])

  (x,y,w,h) = ref_rect
  template = img_ref[y:y+h,x:x+w].copy()
  template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)   
  template = cv2.GaussianBlur(template, (3,3), 0)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template)
  alpha = 255.0/(max_val - min_val)
  beta = -alpha * min_val
  cv2.convertScaleAbs(template, template, alpha, beta)
  edged_template = cv2.Canny(template, low_thresh, high_thresh)

  template_ins = img_ins[y:y+h,x:x+w].copy()
  template_ins = cv2.cvtColor(template_ins, cv2.COLOR_BGR2GRAY)   
  template_ins = cv2.GaussianBlur(template_ins, (3,3), 0)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_ins)
  alpha = 255.0/(max_val - min_val)
  beta = -alpha * min_val
  
  ins_rect = [x+(1.0-search_scale)*w/2, y+(1.0-search_scale)*h/2, search_scale*w, search_scale*h]  
  ins_rect[0] = max(0, ins_rect[0])
  ins_rect[1] = max(0, ins_rect[1])
  xb = min(ins_rect[0] + ins_rect[2] -1.0, width)
  yb = min(ins_rect[1] + ins_rect[3] -1.0, height)
  ins_rect[2] = xb - ins_rect[0] + 1;
  ins_rect[3] = yb - ins_rect[1] + 1;  
  ins_rect = [int(j) for j in ins_rect]
  ins_region = img_ins[ins_rect[1]:(ins_rect[1]+ins_rect[3]),ins_rect[0]:(ins_rect[0]+ins_rect[2])].copy()
  template_ins = cv2.cvtColor(ins_region, cv2.COLOR_BGR2GRAY)   
  ins_region = cv2.GaussianBlur(ins_region, (3,3), 0)
  cv2.convertScaleAbs(ins_region, ins_region, alpha, beta)
  edged = cv2.Canny(ins_region, low_thresh, high_thresh)

  if show:
    # Display the resulting frame
    cv2.imshow('Template',edged_template)
    cv2.imshow('Edged',edged)
    cv2.waitKey(100);

  meth = methods[0]
  method = eval(meth)
  res = cv2.matchTemplate(edged,edged_template,method)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
  else:
    top_left = max_loc
  bottom_right = (top_left[0] + ref_rect[2], top_left[1] + ref_rect[3])
  top_left = (top_left[0]+ins_rect[0], top_left[1]+ins_rect[1])
  bottom_right = (bottom_right[0]+ins_rect[0], bottom_right[1]+ins_rect[1])  
  # update reference roi based on the current position
  ref_rect = [top_left[0], top_left[1], bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]]

  if show:
    cv2.rectangle(frame_ins,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(frame_ins,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
  return True,ref_rect

def cv2ver():
    major = cv2.__version__.split('.')[0]
    return int(major)

if __name__ == '__main__':
  import sys
  cap = cv2.VideoCapture(sys.argv[1])
  # Check if camera opened successfully
  if not cap.isOpened(): 
    print("Error opening video stream or file: "+ sys.argv[1])
    
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  print(frame_width,frame_height)
  if cv2ver() == 3:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); #get the frame count
  else:
    count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)); #get the frame count

  i = 1

  ref_rect = [229, 365, 84, 48] # example

  # Read until video is completed
  while(cap.isOpened()):
    # Capture reference-frame
    if cv2ver() == 3:
      cap.set(cv2.CAP_PROP_POS_FRAMES,i-1); #Set index to last frame
    else:
      cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i-1); #Set index to last frame
    ret, frame_ref = cap.read()
    if not ret:
      break;

    # Capture current-frame         
    if cv2ver() == 3:
      cap.set(cv2.CAP_PROP_POS_FRAMES,i); #Set index to current frame
    else:
      cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i); #Set index to current frame
    ret, frame_ins = cap.read()
    if not ret:
      break;

    img_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)   
    img_ins = cv2.cvtColor(frame_ins, cv2.COLOR_BGR2GRAY)  

    ret,ref_rect = tmplMatch(img_ins,img_ref,ref_rect,frame_width,frame_height,1.6,True)

    i=i+1
    print "i=", i
    print ref_rect
    
  
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    if i == 50:
      break;
 
  # When everything done, release the video capture object
  cap.release()
 
  # Closes all the frames
  cv2.destroyAllWindows()

