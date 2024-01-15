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


chessboardSize = (24, 17)
# [Q] Please discuss the meaning of the paramter
# [A] 실제는 25x 18인데 왜지/

frameSize = (1440, 1080)
# [Q] Please discuss the meaning of the paramter
# [A] 진짜 이미지 사이즈 ㅇㅇ


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# [Q] Please discuss why we need "criteria".
# [A] cornerSubPix에서 종료할 조건(지점)을 정해주기 위해서 필요함.


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm
# [Q] Please discuss the meaning of the paramter. Specify the unit of it.


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
# [Q] How many points? -> 21

imgpoints = []  # 2d points in image plane.
# [Q] How many points? -> 21


images = glob.glob('./samples/*.png')           # [Q] How many images? [A] 21
pdb.set_trace()

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # print(ret, corners)
    # [Q] Please discuss the role of "findChessboardCorners()". Specify the meaning of its outputs.
    # [A]
    # retval : 패턴이 감지 되었는지에 따른 T/F
    # corners : 감지된 코너 개수에 따른 위치(x,y좌표 값)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('imgs', img)
        cv.waitKey(1000)
    # pdb.set_trace()

cv.destroyAllWindows()

# print(len(objpoints), len(imgpoints))
# pdb.set_trace()


############################ CALIBRATION ############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# print(ret, cameraMatrix, dist, rvecs, tvecs, sep='\n')

pdb.set_trace()
'''
    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy).
    [A]
    cameraMatrix는 calibration Matrix로 focal length와 principal point의 정보를 가지고 있다.
    fx = cameraMatrix[0,0], fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2], cy = cameraMatrix[1,2]

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.
    [A]
    Lens distortion coefficients(렌즈 왜곡 계수)에는 2가지 정보를 가지고 있음
    radial distortion coefficients = dist[:3]
    tangential distortion coefficients = dist[3:]

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs".
    [A]
    Rotation Vector, 3x1 회전 벡터, 벡터의 방향은 회전 축을 지정하고 벡터의 크기는 회전 각을 지정함.
    Translation Vector, 3x1 이동 벡터

'''


############################ UNDISTORTION ############################
img_path = './samples/Image__2018-10-05__10-30-08.png'
img_name = img_path.split('__')[-1]
print(img_name)

img = cv.imread(img_path)
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
# print(newCameraMatrix, roi)
'''
    [Q] Please discuss the role of "getOptimalNewCameraMatrix". Why do we need this?
    [A]
    구했던 cameraMatrix를 이용해 보정된 CameraMatrix를 구하는 역할.
    -> 왜곡되어 저장된 사진(행렬)을 올바르게 보정시킴.(왜곡제거)

    [Q] What is "roi"?
    [A]
    region of Interest인줄 알았으나
    Optional output rectangle [x,y,w,h] that outlines all-good-pixels region in the undistorted image.
    보정된 이미지에서 양호한 픽셀들 지역의 아웃라인.
    실제로 이 코드에서는 결과 이미지를 자르는 영역으로 사용된다.

'''

# roi 시각화 해보기
# x, y, w, h = roi
# rect_img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
# cv.imshow('asd', rect_img)
# cv.imwrite('origin_rect.png', rect_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# pdb.set_trace()


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# print(roi)
# print(dst.shape)
cv.imwrite(f'./results/{img_name}_calibResult1.png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5
    )
# print(mapx.shape, mapy.shape)

dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# [Q] Please discuss the role of "remap()"
# [A]
# cv2.initUndistrotRectifyMap : undistortion과 rectification transformation 맵을 계산함. 매핑을 형성?
# cv2.remap : initUndistrotRectifyMap()에서 형성한 매핑 값?에 geometrical transformation을 적용. 최종적으로 왜곡 보정


# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# print(roi)
# print(dst.shape)
cv.imwrite(f'./results/{img_name}_calibResult2.png', dst)

'''
    [Q] Please discuss the difference between 'calibResult1.png' and 'calibResult2.png'.
    [A]
    cv2.undistort로 한번에 왜곡제거를 처리했냐, cv2.initUndistrotRectifyMap -> cv2.remap 2단계로 왜곡을 제거했냐의 차이임.
    cv2.undistort는 initUndistrotRectifyMap과 remap을 합쳐놓은(묶어놓은) 메서드.

    image size도 다르다.
    calibResult1.png는 (1206 x 949), calibResult2.png는 (1121 x 850)이다.
    dst(Destination image) 값이 다르기 때문에
'''
# rectification : Epipolar line을 일치시키는 것.


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
        )
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

'''
    [Q] What is the meaning of "Reprojection Error"? Please specify the unit of it.
    [A]
    계산한 파라미터가 얼마나 정확한지 측정할 수 있는 평가지표. 0에 가까울 수록 정확해짐.

    projectPoints에 intrinsic행렬, rotation행렬, translation행렬, lens distortion행렬을 넣어줌.
    우리가 구한 imgpoints[i]와 projectPoints메서드의 리턴 값인 imgpoints2을 비교해서 L2 norm을 구하고 
    그 값들을 더하여 계산된 에러의 평균을 계산해준다.
    모든 이미지의 에러의 평균 값.
'''

pdb.set_trace()
