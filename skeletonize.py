import camera_calibrate
import numpy as np
import cv2
import sys
#adjust this to reflect where you have tf-openpose
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

def get_skeleton(im, e, scales, shape=None):
    # estimate human poses from a single image !
    image = common.read_imgfile(im, None, None)
    if shape is not None:
        shape.clear()
        shape.extend(image.shape[:-1])
    humans = e.inference(image, scales=scales) #list of skeletons
    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False) #"draws" the skeletons on the image
    return humans

#the path here should point to the directory with the calibration pics
calibration = camera_calibrate.StereoCalibration(r'Z:\intermitent data\frames\\', False)

A_RT_left = calibration.camera_model['M1'].dot(
    np.hstack([calibration.camera_model['R'], calibration.camera_model['T']]))
A_RT_right = calibration.camera_model['M2'].dot(
    np.hstack([calibration.camera_model['R'], calibration.camera_model['T']]))


model = prep_model()
im_shape = []
lhumans = get_skeleton(r'Z:\intermitent data\poses\L_3.png', shape=im_shape, **model)
rhumans = get_skeleton(r'Z:\intermitent data\poses\R_3.png', **model)
lparts = []
rparts = []
for matching_part in list(set(rhumans[0].body_parts.keys()).intersection(lhumans[0].body_parts.keys())):
    #maybe undistort before appending?
    rparts.append([rhumans[0].body_parts[matching_part].x*im_shape[0],
                   rhumans[0].body_parts[matching_part].y*im_shape[1]])
    lparts.append([lhumans[0].body_parts[matching_part].x*im_shape[0],
                   lhumans[0].body_parts[matching_part].y*im_shape[1]])


lparts = np.array(lparts).reshape([2, -1])
rparts = np.array(rparts).reshape([2, -1])
triangulated_4d = cv2.triangulatePoints(A_RT_left, A_RT_right, lparts, rparts).reshape([-1, 4])
triangulated_3d = triangulated_4d[:, :3] / triangulated_4d[:, -1, np.newaxis]
print(triangulated_3d)

#todo investigate https://github.com/markjay4k/3D-Pose-Estimation/blob/master/pt4-webcam3D.py and his use of Prob3dPose
#todo which i think comes from https://github.com/DenisTome/Lifting-from-the-Deep-release