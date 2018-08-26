import numpy as np
from PIL import Image

# desired_shape = [640, 368]
desired_shape = [368, 640]

def resize(im):
    if type(im) == str:
        im = Image.open(im)
    else:
        im = Image.fromarray(im)
    im.thumbnail(desired_shape)
    # im = np.rollaxis(np.array(im), 1)
    im = np.array(im)
    img = np.zeros([*reversed(desired_shape), 3], dtype=np.uint8)
    img[:im.shape[0], :im.shape[1]] = im
    return img
    # return np.pad(im,np.subtract(desired_shape, im.shape[:-1])//2, 'constant')
