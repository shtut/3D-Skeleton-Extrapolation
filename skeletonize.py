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
# import matplotlib.pyplot as plt
import scipy.io as sio

#disable scientific notation
np.set_printoptions(suppress=True)

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
lhumans = get_skeleton(r'Z:\intermitent data\poses\L_pose1.png', shape=im_shape, **model)
rhumans = get_skeleton(r'Z:\intermitent data\poses\R_pose1.png', **model)
lparts = []
rparts = []
indexes = {}
i = 0
for matching_part in list(set(rhumans[0].body_parts.keys()).intersection(lhumans[0].body_parts.keys())):
# for matching_part in [0, 1]:
    #maybe undistort before appending?
    rparts.append([rhumans[0].body_parts[matching_part].x,
                   rhumans[0].body_parts[matching_part].y])
    lparts.append([lhumans[0].body_parts[matching_part].x,
                   lhumans[0].body_parts[matching_part].y])
    k = rhumans[0].body_parts[matching_part].get_part_name()
    indexes[k.value] = i
    i += 1


lparts = np.array(lparts).T #* [[im_shape[0]], [im_shape[1]]]
rparts = np.array(rparts).T #* [[im_shape[0]], [im_shape[1]]]
lparts1 = np.array([lparts[0] * [im_shape[0]], lparts[1]*[im_shape[1]]]).T
rparts1 = np.array([rparts[0] * [im_shape[0]], rparts[1]*[im_shape[1]]]).T

# triangulated_4d = cv2.triangulatePoints(A_RT_left, A_RT_right, lparts, rparts)
# triangulated_4d = cv2.triangulatePoints(np.eye(4)[:3], A_RT_right, lparts, rparts)
# triangulated_4d = triangulated_4d/triangulated_4d[3]
# triangulated_3d = triangulated_4d[:-1].T

# triangulated_3d = triangulated_4d[:, :3] / triangulated_4d[:, -1, np.newaxis]
# print(str(triangulated_3d).replace('[', '').replace(']','').replace('\n  ', '\n')
#       .replace('  ', ' ').replace(' ', ', ')[2:])

#todo investigate https://github.com/markjay4k/3D-Pose-Estimation/blob/master/pt4-webcam3D.py and his use of Prob3dPose
#todo which i think comes from https://github.com/DenisTome/Lifting-from-the-Deep-release


lpoints = []
rpoints = []
for x,y in common.CocoPairsRender:
    if x in indexes and y in indexes:
        lpoints.append(lparts.T[indexes[x]].tolist())
        lpoints.append(lparts.T[indexes[y]].tolist())
        rpoints.append(rparts.T[indexes[x]].tolist())
        rpoints.append(rparts.T[indexes[y]].tolist())

# r = Tk()
# r.withdraw()
# r.clipboard_clear()
# r.clipboard_append(str(rpoints).replace('],', '],\n'))
# r.clipboard_clear()
# r.clipboard_append(str(lpoints).replace('],', '],\n'))
sio.savemat('thingy.m', {'l':lpoints, 'r':rpoints})