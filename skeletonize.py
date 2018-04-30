# Alex Nulam, Dvir Segal and Hadas Shahar [30-Apr-18]
# This python script generates a list of equal skeleton points from 2 images (left and right)
# The points are exported as mat file

import numpy as np
import sys

# adjust this to reflect where you have tf-openpose
sys.path.append(r'../tf-pose-estimation')
sys.path.append(r'../tf-pose-estimation/src')
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import ast
import scipy.io as sio

# disable scientific notation
np.set_printoptions(suppress=True)


def prep_model():
    """
    This method init an OpenPose model
    :return: return a {estimator , scales}
    """
    args = type('', (), {})
    args.resolution = '432x368'
    args.model = 'mobilenet_thin'
    args.scales = '[None]'
    scales = ast.literal_eval(args.scales)
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    return {'e': e, 'scales': scales}


def get_skeleton(im, e, scales, shape=None):
    """
    This method returns a list of skeletons per given image
    :param im: given image
    :param e: estimator (tf-pose estimator)
    :param scales:
    :param shape: image shape
    :return: return a list of skeletons
    """
    # estimate human poses from a single image !
    image = common.read_imgfile(im, None, None)
    if shape is not None:
        shape.clear()
        shape.extend(image.shape[:-1])
    humans = e.inference(image, scales=scales)  # list of skeletons
    return humans


# get the model
model = prep_model()

im_shape = []
# change path to whatever image you would like to skeletonize using OpenPose
left_image_humans = get_skeleton(r'Data\intermitent data\poses\L_pose1.png', shape=im_shape, **model)
# change path to whatever image you would like to skeletonize using OpenPose
right_image_humans = get_skeleton(r'Data\intermitent data\poses\R_pose1.png', **model)

left_image_parts = []
right_image_parts = []
indexes = {}
i = 0
for matching_part in list(
        set(right_image_humans[0].body_parts.keys()).intersection(left_image_humans[0].body_parts.keys())):
    # for matching_part in [0, 1]:
    right_image_parts.append([right_image_humans[0].body_parts[matching_part].x,
                              right_image_humans[0].body_parts[matching_part].y])
    left_image_parts.append([left_image_humans[0].body_parts[matching_part].x,
                             left_image_humans[0].body_parts[matching_part].y])
    k = right_image_humans[0].body_parts[matching_part].get_part_name()
    indexes[k.value] = i
    i += 1

left_image_parts = np.array(left_image_parts).T
right_image_parts = np.array(right_image_parts).T

# Right left and right points to mat file
leftPoints = []
rightPoints = []
for x, y in common.CocoPairsRender:
    if x in indexes and y in indexes:
        leftPoints.append(left_image_parts.T[indexes[x]].tolist())
        leftPoints.append(left_image_parts.T[indexes[y]].tolist())
        rightPoints.append(right_image_parts.T[indexes[x]].tolist())
        rightPoints.append(right_image_parts.T[indexes[y]].tolist())

sio.savemat('lr', {'l': leftPoints, 'r': rightPoints})
