A readme file describing the layout of the code

compare.m:

compare
load kinect data and calculate openpose 3D data
finds the rotation matrix and the scaling factor
finally ,  3D plot all points on viewer

kinect_mapping.py:

get_skeleton
parses the kinect skeleton file to our data structure 
and saves the data as a mat file

skeletonize.m:

matlab script which its purpose is to triangulate the left and right skeletons
generated by OpenPose (using python script)

skeletonize.py:
This python script generates a list of equal skeleton points from 2 images (left and right)
The points are exported as mat file
Then points are used by matlab skeletonize script which triangulate the left and right skeletons from mat file
At the end a 3D skeleton plotted


Skeletonizer
This method init an OpenPose model

get_skeleton
This method returns a list of skeletons per given image
param im: given image
param shape: image shape
return: return a list of skeletons

prepare_op_data
generate a skeleton map for visualization and the join point map

visualize
runs the matlab visualization

get_triangulated_points
turns our 2d point cloud intoa 3d point cloud

save_text
save the cloud in the same format as the kinect text

triangulateOpenpose.m:

triangulateOpenpose(calibration, fig)
given calibration, and a figure containing 2 OpenPose skeletons (2D) - generates a 3D OpenPose skeleton
calibration - name of the calibraion file containing matrices for the 2 cameras (stereo)
fig - name of the text file containing both skeletons

rigidTransform3D.m
This function finds the optimal Rigid/Euclidean transform in 3D space
It expects as input a Nx3 matrix of 3D points.
It returns R, t

folder_processor.py
loops over two given folder content and traingulate between each image pair.
The assumption that images are ordered equally, it generates a 3D skeleton per image pair and saves it to disk
