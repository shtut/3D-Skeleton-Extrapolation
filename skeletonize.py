# Alex Nulman, Dvir Segal and Hadas Shahar [30-Apr-18]
# This python script generates a list of equal skeleton points from 2 images (left and right)
# The points are exported as mat file

# Then points are used by matlab skeletonize script which triangulate the left and right skeletons from mat file
# At the end a 3D skeleton plotted

import numpy as np
import sys

import ast
import scipy.io as sio
import kinect_mapping

# adjust this to reflect where you have tf-openpose
# sys.path.append(r'../tf-openpose')
sys.path.append(r'../tf-openpose/src')
# sys.path.append(r'tf-openpose')
sys.path.append(r'tf-openpose/src')
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import matlab.engine
# disable scientific notation
np.set_printoptions(suppress=True)

# this line starts matlab in the background so it takes a while
eng = matlab.engine.start_matlab()

class Skeletonizer(object):
    def __init__(self, calibration_file):
        """
        This method init an OpenPose model
        """
        args = type('', (), {})
        args.resolution = '432x368'
        args.model = 'mobilenet_thin'
        args.scales = '[None]'
        scales = ast.literal_eval(args.scales)
        w, h = model_wh(args.resolution)
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        self.model = {'e': e, 'scales': scales}
        # start new matlab session
        self.matlab = matlab.engine.start_matlab()
        # or...
        # in matlab run: matlab.engine.shareEngine
        # and uncomment the following line
        # self.matlab = matlab.engine.connect_matlab()
        self.calibration = calibration_file

    def __del__(self):
        if self.matlab is not None:
            self.matlab.exit()

    def close(self):
        self.__del__()

    def get_skeleton(self, im, shape=None):
        """
        This method returns a list of skeletons per given image
        :param im: given image
        :param shape: image shape
        :return: return a list of skeletons
        """
        # estimate human poses from a single image !
        image = common.read_imgfile(im, None, None)
        if shape is not None:
            shape.clear()
            shape.extend(image.shape[:-1])
        humans = self.model['e'].inference(image, scales=self.model['scales'])  # list of skeletons
        return humans

    def prepare_op_date(self, im1, im2, output_name='lr'):
        global right_image_parts, left_image_parts, indexes, right_image_humans
        im_shape = []
        left_image_humans = self.get_skeleton(im1, shape=im_shape)
        right_image_humans = self.get_skeleton(im2)

        left_image_parts = []
        right_image_parts = []
        indexes = {}
        i = 0
        anchorsl = []
        anchorsr = []
        part_order = []
        part_order_vis = []
        for matching_part in list(
                set(right_image_humans[0].body_parts.keys()).intersection(left_image_humans[0].body_parts.keys())):
            # for matching_part in [0, 1]:
            right_image_parts.append([right_image_humans[0].body_parts[matching_part].x,
                                      right_image_humans[0].body_parts[matching_part].y])
            left_image_parts.append([left_image_humans[0].body_parts[matching_part].x,
                                     left_image_humans[0].body_parts[matching_part].y])
            k = right_image_humans[0].body_parts[matching_part].get_part_name()
            indexes[k.value] = i
            part_order.append(k.name)
            i += 1

        for i in [2, 5, 8, 11]:  # shoulders and hips
            try:
                anchorsl.append([right_image_humans[0].body_parts[i].x,
                                 right_image_humans[0].body_parts[i].y])
                anchorsr.append([left_image_humans[0].body_parts[i].x,
                                 left_image_humans[0].body_parts[i].y])
            except Exception as e:
                print('missing  anchor points, cannot process this image.')
                raise e

        left_image_parts = np.array(left_image_parts)
        right_image_parts = np.array(right_image_parts)
        anchorsl = np.array(anchorsl).tolist()
        anchorsr = np.array(anchorsr).tolist()

        # Right left and right points to mat file
        left_points = []
        right_points = []
        for x, y in common.CocoPairsRender:
            if x in indexes and y in indexes:
                left_points.append(left_image_parts[indexes[x]].tolist())
                left_points.append(left_image_parts[indexes[y]].tolist())
                right_points.append(right_image_parts[indexes[x]].tolist())
                right_points.append(right_image_parts[indexes[y]].tolist())
                part_order_vis.append(common.CocoPart(indexes[x]).name)
                part_order_vis.append(common.CocoPart(indexes[y]).name)

        sio.savemat(output_name + '_vis', {'l': left_points, 'r': right_points, 'al': anchorsl, 'ar': anchorsr,
                                           'order': part_order_vis})
        sio.savemat(output_name, {'l': left_image_parts, 'r': right_image_parts, 'order':part_order})
        self.part_order = part_order
        return output_name + '_vis', output_name

    def visualize(self, open_pose_skeleton_mat, kinect_skeleton_file):
        self.matlab.compare(self.calibration, open_pose_skeleton_mat, kinect_skeleton_file, nargout=0)

    def get_triangulated_points(self, point_file):
        return self.matlab.triangulateOpenpose(self.calibration, point_file)

    def save_text(self, triangulated_points, out_file):
        with open(out_file, 'w') as fh:
            for line,part in zip(triangulated_points,self.part_order):
                fh.write('#'.join(list(map(str,line)) + [part]))
                fh.write('\n')


# if __name__ == '__main__':
im1 = r'Z:\3D_Skeleton_Extraction\new_kinect_data\poses\frame1312.png'
im2 = r'Z:\3D_Skeleton_Extraction\new_kinect_data\poses\frame1728.png'
kinect_text = r'Z:\3D_Skeleton_Extraction\new_kinect_data\poses\-8586705244807568198%940194864624.TXT'
calibration = 'calibration.mat'
op_out_file = 'lr.mat'
sk = Skeletonizer(calibration)
file_for_vis, point_file = sk.prepare_op_date(im1, im2)
kin_out_file = kinect_mapping.get_skeleton(kinect_text)
sk.visualize(file_for_vis, kin_out_file)
triangulated_points = sk.get_triangulated_points(point_file)
sk.save_text(triangulated_points, 'skeleton_dump.txt')

