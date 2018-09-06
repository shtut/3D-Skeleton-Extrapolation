# Alex Nulman, Dvir Segal and Hadas Shahar [05-Sep-2018]
# loops over two given folder content and traingulate between each image pair.
# The assumption that images are ordered equally, it generates a 3D skeleton per image pair and saves it to disk

import argparse
from glob import glob
from os.path import basename, exists, realpath
from os import makedirs
import skeletonize

parser = argparse.ArgumentParser(description='skeletonize directories')
parser.add_argument('--calibration', type=str, required=True, help='path to calibration file')
parser.add_argument('--dir1', type=str, required=True, help='path to directory containing first set of images')
parser.add_argument('--dir2', type=str, required=True, help='path to directory containing second set of images')
parser.add_argument('--out', type=str, default='results', help='path to output directory')
args = parser.parse_args()

print("starting skeletonizer....", end='')
sk = skeletonize.Skeletonizer(args.calibration)
print('done.')

def process_image_pair(im1, im2):
    _, point_file = sk.prepare_op_data(im1, im2)
    triangulated_points = sk.get_triangulated_points(point_file)
    out_path = args.out + '/' + (basename(im1)[:-3]) + 'txt'
    # print(out_path)
    sk.save_text(triangulated_points, out_path)

# print(args.dir1, args.dir2)
files1 = glob(args.dir1 + '/*[jpg,png]')
files2 = glob(args.dir2 + '/*[jpg,png]')
if not exists(args.out):
    makedirs(args.out)
# print(files1, files2)
for im1, im2 in zip(files1, files2):
    process_image_pair(im1, im2)
