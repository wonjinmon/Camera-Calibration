'''
    Seokju Lee @ EE6102
    2023.10.30.

    Original code from 
    https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
    and
    https://github.com/niconielsen32/ComputerVision

    Korean reference
    https://leechamin.tistory.com/345

    Prerequisite:
    which pip
    pip install opencv-contrib-python

    If you want to experience VirtualCam, please follow the below instruction.
    https://github.com/kaustubh-sadekar/VirtualCam
'''

import glob
import pdb

import numpy as np
import cv2 as cv


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ################


chessboardSize = (24, 17)   # [Q] Please discuss the meaning of the paramter
frameSize = (1440, 1080)    # [Q] Please discuss the meaning of the paramter



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # [Q] Please discuss why we need "criteria".


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm     # [Q] Please discuss the meaning of the paramter. Specify the unit of it.


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space  # [Q] How many points?
imgpoints = []  # 2d points in image plane.     # [Q] How many points?


images = glob.glob('./samples/*.png')           # [Q] How many images?
# pdb.set_trace()

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)     # [Q] Please discuss the role of "findChessboardCorners()". Specify the meaning of its outputs.

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
    # pdb.set_trace()

cv.destroyAllWindows()
# pdb.set_trace()



############################ CALIBRATION ############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# pdb.set_trace()
'''
    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy).

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs".

'''


############################ UNDISTORTION ############################

img = cv.imread('./samples/Image__2018-10-05__10-36-33.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
'''
    [Q] Please discuss the role of "getOptimalNewCameraMatrix". Why do we need this?

    [Q] What is roi"?

'''


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./results/calibResult1.png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)    # [Q] Please discuss the role of "remap()"


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./results/calibResult2.png', dst)

'''
    [Q] Please discuss the difference between 'calibResult1.png' and 'calibResult2.png'.
'''


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

'''
    [Q] What is the meaning of "Reprojection Error"? Please specify the unit of it.
'''

# pdb.set_trace()
