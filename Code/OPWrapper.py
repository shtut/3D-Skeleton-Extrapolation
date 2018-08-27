import numpy as np
import sys
import ast
sys.path.append(r'E:\cygwin64\home\Alex\git\tf-openpose\src')
sys.path.append(r'E:\cygwin64\home\Alex\git\tf-openpose')
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import matlab.engine

args = type('', (), {})
resolutions = ['432x368','656x368','1312x736','1920x1088']
args.model = 'mobilenet_thin'
max_joints = []
for res in resolutions:
# args.resolution = '1312x736'
    args.resolution = res
    args.scales = '[None]'
    scales = ast.literal_eval(args.scales)
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    model = {'e': e, 'scales': scales}
