import cv2

def splitter(vid_path, side, out_path):
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    count = 0
    # success = True
    while success:
        success, image = vidcap.read()
        cv2.imwrite("{}\\{}_{}.jpg".format(out_path, side, count), image)
        count += 1