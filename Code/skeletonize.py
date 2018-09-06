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
import cv2

# adjust this to reflect where you have tf-openpose
sys.path.append(r'../tf-openpose/src')
sys.path.append(r'../../tf-openpose/src')
sys.path.append(r'../../../tf-openpose/src')
sys.path.append(r'./tf-openpose/src')

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import matlab.engine
# disable scientific notation
np.set_printoptions(suppress=True)


class Skeletonizer(object):
    def __init__(self, calibration_file):
        """
        This method init an OpenPose model
        """
        args = type('', (), {})
        # args.resolution = '1312x736'
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
        '''destructor. closes the matlab session'''
        if self.matlab is not None:
            self.matlab.exit()

    def close(self):
        '''closes the matlab session'''
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

    def prepare_op_data(self, im1, im2, output_name='OpenPoseMat'):
        '''
        turns a pair of images into a skeleton file for visualisation and a joint cloud for computational use
        and saves them as mat files
        :param im1: path to image
        :param im2: path to image
        :param output_name: file name for the output file
        :return: filename of the output for visualisation, filename of point cloud
        '''
        global right_image_parts, left_image_parts, indexes, right_image_humans
        im_shape = []
        left_image_humans = self.get_skeleton(im1, shape=im_shape)
        right_image_humans = self.get_skeleton(im2)

        left_image_parts = []
        right_image_parts = []
        indexes = {}
        i = 0

        # assuming im1 and im2 are the same size
        height,width,_ = cv2.imread(im1).shape

        part_order = []
        part_order_vis = []
        for matching_part in list(
                set(right_image_humans[0].body_parts.keys()).intersection(left_image_humans[0].body_parts.keys())):
            # for matching_part in [0, 1]:
            right_image_parts.append([right_image_humans[0].body_parts[matching_part].x * width/1000,
                                      right_image_humans[0].body_parts[matching_part].y * height/1000])
            left_image_parts.append([left_image_humans[0].body_parts[matching_part].x * width/1000,
                                     left_image_humans[0].body_parts[matching_part].y * height/1000])
            k = right_image_humans[0].body_parts[matching_part].get_part_name()
            indexes[k.value] = i
            part_order.append(k.name)
            i += 1


        left_image_parts = np.array(left_image_parts)
        right_image_parts = np.array(right_image_parts)


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

        sio.savemat(output_name + '_vis', {'l': left_points, 'r': right_points,  'order': part_order_vis})
        sio.savemat(output_name, {'l': left_image_parts, 'r': right_image_parts, 'order':part_order})
        self.part_order = part_order
        return output_name + '_vis', output_name

    def visualize(self, open_pose_skeleton_mat, kinect_skeleton_file):
        '''
        run the matlab compare script and draws the skeletons
        :param open_pose_skeleton_mat: path to openpose skeleton mat file
        :param kinect_skeleton_file: path to kinect skeleton mat file
        '''
        self.matlab.compare(self.calibration, open_pose_skeleton_mat, kinect_skeleton_file, nargout=0)

    def get_triangulated_points(self, point_file):
        '''
        triangulates an openpose skeleton mat into a 3d skeleton
        :param point_file: path to 2d openpose skeleton mat
        :return: 3d skeleton points
        '''
        return self.matlab.triangulateOpenpose(self.calibration, point_file)

    def save_text(self, triangulated_points, out_file):
        '''
        parses a 3d openpose skeleton into a text file in a similar format to the kinect skeleton text
        :param triangulated_points: list of points (as generated by get_triangulated_points)
        :param out_file: name of output file
        '''
        with open(out_file, 'w') as fh:
            for line,part in zip(triangulated_points,self.part_order):
                fh.write('#'.join(list(map(str,line)) + [part]))
                fh.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='skeletonize demo')
    parser.add_argument('--calibration', type=str, help='path to calibration file',
                        default='example/aug17_stereoParams.mat')
    parser.add_argument('--img1', type=str, help='path to first image file',
                        default=r'../data/Original/Test/-8586670814778381942%1458000246174__.png')
    parser.add_argument('--img2', type=str, help='path to second image file (the has a kinect text file)',
                        default=r'../data/Original/Test/-8586670814788377494%2226568669020__.png')
    parser.add_argument('--kinect', type=str, help='path to kinect text file matching the 2nd image',
                        default=r'../data/Original/Test/-8586670814788377494%2226568606520.TXT')
    parser.add_argument('--out', type=str, default='skeleton_dump', help='output file name')

    args = parser.parse_args()

    #initilize our class and matlab engine (runs headless matlab)
    sk = Skeletonizer(args.calibration)
    #generate a skeleton map for visualization and the join point map
    file_for_vis, point_file = sk.prepare_op_data(args.img1, args.img2)
    #parse the kinect skeleton text into our data format
    kin_out_file = kinect_mapping.get_skeleton(args.kinect)
    #runs the matlab visualization
    sk.visualize(file_for_vis, kin_out_file)
    #turns our 2d point cloud intoa 3d point cloud
    triangulated_points = sk.get_triangulated_points(point_file)
    # save the cloud in the same format as the kinect text
    sk.save_text(triangulated_points, args.out + '.txt')
    input("Press Enter to continue...")
