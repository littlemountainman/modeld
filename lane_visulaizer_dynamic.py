from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from common.lanes_image_space import transform_points
import os
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import cv2
import sys

#matplotlib.use('Agg')
camerafile = sys.argv[1]
supercombo = load_model('models/supercombo.keras')

#print(supercombo.summary())

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)

x_left = x_right = x_path = np.linspace(0, 192, 192)
(ret, previous_frame) = cap.read()
if not ret:
   exit()
else:
  img_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[0] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))

while True:
  plt.clf()
  plt.title("lanes and path")
  plt.xlim(0, 1200)
  plt.ylim(800, 0)
  (ret, current_frame) = cap.read()
  if not ret:
       break
  frame = current_frame.copy()
  img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[1] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
  frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0

  inputs = [np.vstack(frame_tensors[0:2])[None], desire, state]
  outs = supercombo.predict(inputs)
  parsed = parser(outs)
  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]   # For 6 DoF Callibration
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  plt.imshow(frame)
  new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
  new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
  new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
  plt.plot(new_x_left, new_y_left, label='transformed', color='w')
  plt.plot(new_x_right, new_y_right, label='transformed', color='w')
  plt.plot(new_x_path, new_y_path, label='transformed', color='green')
  imgs_med_model[0]=imgs_med_model[1]
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#plt.show()
  


