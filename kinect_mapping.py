import numpy as np
import scipy.io as sio

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


def get_skeleton(path, output='other1'):
    d = {}
    l = []
    point_cloud = []
    fh = open(path)
    for line in fh:
        stuff = line.split('#')
        point_cloud.append(list(map(float, stuff[:3])))
        d[stuff[-1].strip()] = np.array(point_cloud[-1])

    if 'Head' in d:
        offset = d['Neck']
    else:
        offset = np.array([0, 0, 0])
    part_order = []
    for x, y in mapping:
        if x in d and y in d:
            l.append(d[x] - offset)
            l.append(d[y] - offset)
            part_order.append(x)
            part_order.append(y)
    anchors = []
    for i in ['ShoulderRight', 'ShoulderLeft', 'HipRight', 'HipLeft']:
        try:
            anchors.append(d[i] - offset)
        except Exception as e:
            print('missing  anchor points, cannot process this image.')
            raise e
    sio.savemat(output, {'both': l, 'anchors': anchors, 'points': point_cloud, 'order': part_order})
    return output+'.mat'