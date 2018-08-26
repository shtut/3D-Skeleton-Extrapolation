# Alex Nulman, Dvir Segal and Hadas Shahar [30-Apr-18]
# Helper function that parses kinect skeleton text files into joint point cloud and skeleton, and saves it as a mat file



import numpy as np
import scipy.io as sio

# parses the kinect skeleton file to our data structure 
mapping = [
    ('Head', 'Neck'),
    ('SpineShoulder', 'ShoulderLeft'),
    ('SpineShoulder', 'ShoulderRight'),
    ('SpineShoulder', 'SpineMid'),
    ('Neck', 'SpineShoulder'),
    ('ShoulderRight', 'ElbowRight'),
    ('ElbowRight', 'WristRight'),
    ('WristRight', 'HandRight'),
    ('WristRight', 'ThumbRight'),
    ('HandRight', 'HandTipRight'),
    ('ShoulderLeft', 'ElbowLeft'),
    ('ElbowLeft', 'WristLeft'),
    ('WristLeft', 'HandLeft'),
    ('WristLeft', 'ThumbLeft'),
    ('HandLeft', 'HandTipLeft'),
    ('SpineMid', 'SpineBase'),
    ('SpineBase', 'HipRight'),
    ('HipRight', 'KneeRight'),
    ('KneeRight', 'AnkleRight'),
    ('AnkleRight', 'FootRight'),
    ('SpineBase', 'HipLeft'),
    ('HipLeft', 'KneeLeft'),
    ('KneeLeft', 'AnkleLeft'),
    ('AnkleLeft', 'FootLeft'),
]


def get_skeleton(path, output='KinectMat'):
    name_to_coor_dict = {}
    skeleton_lines = []
    joints = []
    fh = open(path)
    # parses the kinect skeleton file to a dictionary {joint: [x,y,z]}
    for line in fh:
        stuff = line.split('#')
        joints.append(list(map(float, stuff[:3])))
        name_to_coor_dict[stuff[-1].strip()] = np.array(joints[-1])
    #move the entire skeleton by the neck point to 0,0,0
    if 'Neck' in name_to_coor_dict:
        offset = name_to_coor_dict['Neck']
    else:
        offset = np.array([0, 0, 0])
    # creates the skeleton_lines list
    part_order = []

    for x, y in mapping:
        if x in name_to_coor_dict and y in name_to_coor_dict:
            skeleton_lines.append(name_to_coor_dict[x] - offset)
            skeleton_lines.append(name_to_coor_dict[y] - offset)
            part_order.append(x)
            part_order.append(y)
    # saves the data as a mat file
    sio.savemat(output, {'lines': skeleton_lines, 'joints': joints, 'order': part_order})
    return output+'.mat'