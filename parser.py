from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
from tools.lib.framereader import FrameReader
import matplotlib
import matplotlib.pyplot as plt
from selfdrive.modeld.constants import LANE_OFFSET, LEAD_X_SCALE, LEAD_Y_SCALE
import cv2 
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softplus(x):
  # fix numerical stability
  #return np.log1p(np.exp(x))
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

def softmax(x):
  x = np.copy(x)
  axis = 1 if len(x.shape) > 1 else 0
  x -= np.max(x, axis=axis, keepdims=True)
  if x.dtype == np.float32 or x.dtype == np.float64:
    np.exp(x, out=x)
  else:
    x = np.exp(x)
  x /= np.sum(x, axis=axis, keepdims=True)
  return x

fr = FrameReader("leon.hevc")
cap = cv2.VideoCapture("leon.hevc")

# Images in yuv are being stored here
imgs = []
raw_imgs = []
for i in tqdm(range(1000)):
  imgs.append(fr.get(i, pix_fmt='yuv420p')[0].reshape((874*3//2, 1164)))
  ret, frame = cap.read()
  raw_imgs.append(frame)

raw_imgs = np.array(raw_imgs)
print(raw_imgs.shape)
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

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
def parser(outs):
    PATH_DISTANCE = 192  
    out_dict = {}
    path, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose = outs
    old_scale = True
    if path is not None:
        if path.shape[1] == PATH_DISTANCE*2 + 1:
            out_dict['path'] = path[:, :PATH_DISTANCE]
            out_dict['path_stds'] = softplus(path[:, PATH_DISTANCE:2*PATH_DISTANCE])
            out_dict['path_stds'][int(path[0,-1]):] = 1e3
    elif path.shape[1] == PATH_DISTANCE*2:
        out_dict['path'] = path[:, :PATH_DISTANCE]
        out_dict['path_stds'] = softplus(path[:, PATH_DISTANCE:2*PATH_DISTANCE])
    else:
        path_reshaped = path[:,:-1].reshape((path.shape[0], -1, PATH_DISTANCE*2 + 1))
        out_dict['paths'] = path_reshaped[:, :, :PATH_DISTANCE]
        out_dict['paths_stds'] = softplus(path_reshaped[:, :, PATH_DISTANCE:PATH_DISTANCE*2])
        out_dict['path_weights'] = softmax(path_reshaped[:,:,-1])
        lidx = np.argmax(out_dict['path_weights'])
        out_dict['path'] = path_reshaped[:, lidx, :PATH_DISTANCE]
        out_dict['path_stds'] = softplus(path_reshaped[:, lidx, PATH_DISTANCE:PATH_DISTANCE*2])


    if ll is not None:
        out_dict['lll'] = ll[:, :PATH_DISTANCE] + LANE_OFFSET
        out_dict['lll_prob'] = sigmoid(ll[:, -1])
        out_dict['lll_stds'] = softplus(ll[:, PATH_DISTANCE:-2])
        out_dict['lll_stds'][int(ll[0,-2]):] = 1e3
    if rl is not None:
        out_dict['rll'] = rl[:, :PATH_DISTANCE] - LANE_OFFSET
        out_dict['rll_prob'] = sigmoid(rl[:, -1])
        out_dict['rll_stds'] = softplus(rl[:, PATH_DISTANCE:-2])
        out_dict['rll_stds'][int(rl[0,-2]):] = 1e3
    if lead is not None:
        if old_scale:
            LEAD_X_SCALE = 140
            LEAD_Y_SCALE = 10
            LEAD_V_SCALE = 10
    else:
        LEAD_V_SCALE = 1
        LEAD_X_SCALE = 10
        LEAD_Y_SCALE = 10

    # LEAD MDN
    lead_reshaped = lead[:,:-3].reshape((-1,5,11))
    lead_weights = softmax(lead_reshaped[:,:,8])
    lidx = np.argmax(lead_weights[0])
    out_dict['lead_xyva'] = np.column_stack([lead_reshaped[:,lidx, 0] * LEAD_X_SCALE,
                                             lead_reshaped[:,lidx, 1] * LEAD_Y_SCALE,
                                             lead_reshaped[:,lidx, 2] * LEAD_V_SCALE,
                                             lead_reshaped[:,lidx, 3]])
    out_dict['lead_xyva_std'] = np.column_stack([softplus(lead_reshaped[:,lidx, 4]) * LEAD_X_SCALE,
                                                 softplus(lead_reshaped[:,lidx, 5]) * LEAD_Y_SCALE,
                                                 softplus(lead_reshaped[:,lidx, 6]) * LEAD_V_SCALE,
                                                 softplus(lead_reshaped[:,lidx, 7])])
    out_dict['lead_prob'] = sigmoid(lead[:, -3])
    lead_weights_2s = softmax(lead_reshaped[:,:,9])
    lidx = np.argmax(lead_weights_2s[0])
    out_dict['lead_xyva_2s'] = np.column_stack([lead_reshaped[:,lidx, 0] * LEAD_X_SCALE,
                                                lead_reshaped[:,lidx, 1] * LEAD_Y_SCALE,
                                                lead_reshaped[:,lidx, 2] * LEAD_V_SCALE,
                                                lead_reshaped[:,lidx, 3]])
    out_dict['lead_xyva_std_2s'] = np.column_stack([softplus(lead_reshaped[:,lidx, 4]) * LEAD_X_SCALE,
                                                    softplus(lead_reshaped[:,lidx, 5]) * LEAD_Y_SCALE,
                                                    softplus(lead_reshaped[:,lidx, 6]) * LEAD_V_SCALE,
                                                    softplus(lead_reshaped[:,lidx, 7])])
    out_dict['lead_prob_2s'] = sigmoid(lead[:, -2])
    out_dict['lead_all'] = lead
    """
    if speed is not None:
        out_dict['speed'] = speed
    """
    if meta is not None:
        out_dict['meta'] = meta
    if desire_pred is not None:
        out_dict['desire'] = desire_pred
    if desire_state is not None:
        out_dict['desire_state'] = desire_state
    
    if long_x is not None:
        out_dict['long_x'] = long_x
    if long_v is not None:
        out_dict['long_v'] = long_v
    if long_a is not None:
        out_dict['long_a'] = long_a
    if pose is not None:
        out_dict['trans'] = pose[:,:3]
        out_dict['trans_std'] = softplus(pose[:,6:9]) + 1e-6

        out_dict['rot'] = pose[:,3:6] * np.pi / 180.0
        out_dict['rot_std'] = (softplus(pose[:,9:12]) + 1e-6) * np.pi / 180.0
    return out_dict
from tensorflow.keras.models import load_model
supercombo = load_model('supercombo.keras')
print(supercombo.summary())
# Just passing zeros for desire and state
poses = []
state = np.zeros((1,512))
## Saving lane data 
left_lane = []
right_lane = []
count = 0
for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], np.zeros((1,8)), state]
  outs = supercombo.predict(inputs)
  parsed = parser(outs)
  #print(parsed)
  poses.append(outs[-2])
  state = outs[-1]
  left_lane.append(outs[1])
  right_lane.append(outs[2])
  ll = parsed["lll"]
  llstd = parsed["lll_stds"]
  cv2.imshow("modeld", raw_imgs[i])
  #if count == 950:
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  plt.clf()
  plt.scatter(ll, range(0,192), c="b")
  plt.scatter(parsed["rll"], range(0, 192), c="r")
  plt.scatter(parsed["path"], range(0, 192), c="g")
  #plt.gca().invert_xaxis()
  plt.pause(0.05)
  count += 1
plt.show()
  


