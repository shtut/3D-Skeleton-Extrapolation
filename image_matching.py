import os
import cv2
import numpy as np

points = np.array([0, 0], dtype='float32')


def main(first_folder, second_folder):
    global points
    first_file = get_all_files(first_folder, 200)
    second_file = get_all_files(second_folder, 400)
    firstImage = cv2.imread(first_file)
    cv2.namedWindow('Source Image')
    cv2.setMouseCallback('Source Image', onmouse)
    cv2.imshow('Source Image', firstImage)
    cv2.waitKey(0)

    # pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    pts_src = np.copy(points)
    points = np.array([0, 0], dtype='float32')

    secondImage = cv2.imread(second_file)
    cv2.namedWindow('Destination Image')
    cv2.setMouseCallback('Destination Image', onmouse)
    cv2.imshow('Destination Image', secondImage)
    cv2.waitKey(0)
    # pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    pts_dst = np.copy(points)

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    # im_out = cv2.warpPerspective(firstImage, h, (secondImage.shape[1], secondImage.shape[0]))

    # Display images
    # cv2.imshow("Source Image", firstImage)
    # cv2.imshow("Destination Image", secondImage)
    # cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)


def onmouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and points.size < 7:
        points = np.vstack([points, np.hstack([x, y])])
    elif event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()


def get_all_files(directoryName, location):
    filePath = os.path.abspath(directoryName)
    fname = os.listdir(filePath)[location]
    filePathWithSlash = filePath + "\\"
    filenameWithPath = os.path.join(filePathWithSlash, fname)
    return filenameWithPath


if __name__ == "__main__":
    main(os.path.abspath(
        "C:\\Users\\300132604\\Desktop\\RGB + Skeleton for Hagit\\Session 1\\Recordings\\Mac\\Session-hagit1\\Color"),
        os.path.abspath(
            "C:\\Users\\300132604\\Desktop\RGB + Skeleton for Hagit\\Session 2\\Recordings\\Mac\\Session-hagit2\\Color"))
