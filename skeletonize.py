import camera_calibrate
import numpy as np
import cv2
import sys
sys.path.append('E:\\cygwin64\\home\\Alex\\git\\tf-openpose')
sys.path.append('E:\\cygwin64\\home\\Alex\\git\\tf-openpose\\src')
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import ast

def prep_model():
    args = type('', (), {})
    args.resolution = '432x368'
    args.model = 'mobilenet_thin'
    args.scales = '[None]'
    scales = ast.literal_eval(args.scales)
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    return {'e': e, 'scales': scales}

def get_skeleton(im, e, scales):
    # estimate human poses from a single image !
    image = common.read_imgfile(im, None, None)
    humans = e.inference(image, scales=scales) #list of skeletons
    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False) #"draws" the skeletons on the image
    return humans

calibration = camera_calibrate.StereoCalibration(r'Z:\intermitent data\frames\\', False)

A_RT_left = calibration.camera_model['M1'].dot(
    np.hstack([calibration.camera_model['R'], calibration.camera_model['T']]))
A_RT_right = calibration.camera_model['M2'].dot(
    np.hstack([calibration.camera_model['R'], calibration.camera_model['T']]))


model = prep_model()
lhumans = get_skeleton(r'Z:\intermitent data\poses\L_2.png', **model)
rhumans = get_skeleton(r'Z:\intermitent data\poses\R_2.png', **model)
lparts = []
rparts = []
for matching_part in list(set(rhumans[0].body_parts.keys()).intersection(lhumans[0].body_parts.keys())):
    rparts.append([rhumans[0].body_parts[matching_part].x, rhumans[0].body_parts[matching_part].y])
    lparts.append([lhumans[0].body_parts[matching_part].x, lhumans[0].body_parts[matching_part].y])

lparts = np.array(lparts).reshape([2, -1])
rparts = np.array(rparts).reshape([2, -1])
cv2.triangulatePoints(A_RT_left, A_RT_right, lparts, rparts)