#! /usr/bin/env python

import mediapipe as mp
import numpy as np
import imutils
import cv2

mpSegment = mp.solutions.selfie_segmentation
mpPose = mp.solutions.pose
mpHolistic = mp.solutions.holistic
mpDraw = mp.solutions.drawing_utils

CHECKERBOARD_ROWS = 9
CHECKERBOARD_COLS = 7
CHECKERBOARD_SIDE_MM = 20
MM_PER_INCH = 25.4

SELFIE_SEGMENTATION_CUTOFF_PROBABILITY = 0.95

CHEST_LINE_RATIO = 0.25
STOMACH_LINE_RATIO = 0.5
WAIST_LINE_RATIO = 0.9

READ_ERROR = 1
PROCESS_ERROR = 2

HEURISTICS = {
    'shoulder': 3,
    'arms': 0,
    'pantsLength': 3,
    'shirtLength': 3,
    'chest': 0,
    'stomach': 0,
    'waist': 0,
}

# pose detection
def getPoseLandmarks(image):
    pose = mpPose.Pose(static_image_mode=True, model_complexity=2)
    results = pose.process(image)
    
    if results.pose_landmarks:
        return results.pose_landmarks
    return None


# selfie segmentation
def getSelfieSegmentationMask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mpSegment.SelfieSegmentation(model_selection=0) as segmentation:
        results = segmentation.process(image)
        return results.segmentation_mask


def findScaleFactor(image):
    # generate object points
    objp = np.zeros((CHECKERBOARD_COLS*CHECKERBOARD_ROWS,3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_ROWS,0:CHECKERBOARD_COLS].T.reshape(-1,2)
    objp = objp * CHECKERBOARD_SIDE_MM

    # find image points
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

    # if checkerboard found
    if ret:
        # find average x and y scales
        x_factors = []
        y_factors = []
        for i in range(1, corners.shape[0]):
            # scale x
            if objp[i][0] - objp[i-1][0] != 0:
                diff_o = objp[i][0] - objp[i-1][0]
                diff_i = corners[i, 0, 0] - corners[i-1, 0, 0]
                x_factors.append(diff_o/diff_i)
            if objp[i][1] - objp[i-1][1] != 0:
                diff_o = objp[i][1] - objp[i-1][1]
                diff_i = corners[i, 0, 1] - corners[i-1, 0, 1]
                y_factors.append(diff_o/diff_i)
        scale_x, scale_y = np.mean(x_factors), np.mean(y_factors)

        # find projection error
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)
        corners2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
        error = cv2.norm(corners, corners2, cv2.NORM_L2) / len(corners2)
        # report
        return (scale_x, scale_y), error
    
    # no checkerboard found
    return None


# convert landmark to world point
def landmarkToWorldPoint(image, landmark, scale_factor):
    h, w, _ = image.shape
    world_point = (landmark.x * w * scale_factor[0], landmark.y * h * scale_factor[1], landmark.z * w * scale_factor[0])
    return np.array(world_point)


# get shoulder line
def shoulderLine(image, landmarks):
    l_shoulder = landmarks.landmark[mpHolistic.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks.landmark[mpHolistic.PoseLandmark.RIGHT_SHOULDER]
    y = np.mean((l_shoulder.y * image.shape[0], r_shoulder.y * image.shape[0]))
    return y


# get hip line
def hipLine(image, landmarks):
    l_hip = landmarks.landmark[mpHolistic.PoseLandmark.LEFT_HIP]
    r_hip = landmarks.landmark[mpHolistic.PoseLandmark.RIGHT_HIP]
    y = np.mean((l_hip.y * image.shape[0], r_hip.y * image.shape[0]))
    return y


# find distance between two points
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# find one side's axes for elliptical measurements
def find_axes(image, hip_line, shoulder_line, scale):
    # determine cuts
    y1 = int(shoulder_line + abs(shoulder_line - hip_line) * CHEST_LINE_RATIO)
    y2 = int(shoulder_line + abs(shoulder_line - hip_line) * STOMACH_LINE_RATIO)
    y3 = int(shoulder_line + abs(shoulder_line - hip_line) * WAIST_LINE_RATIO)
    l1, r1 = None, None
    l2, r2 = None, None
    l3, r3 = None, None

    # find left and right points
    mask = getSelfieSegmentationMask(image)
    for x in range(image.shape[1]):
        # chest
        if l1 is None:
            if mask[y1, x] > SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
                l1 = (x, y1)
        elif r1 is None:
            if mask[y1, x] < SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
               r1 = (x, y1)
        
        # stomach
        if l2 is None:
            if mask[y2, x] > SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
                l2 = (x, y2)
        elif r2 is None:
            if mask[y2, x] < SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
                r2 = (x, y2)
        
        # waist
        if l3 is None:
            if mask[y3, x] > SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
                l3 = (x, y3)
        elif r3 is None:
            if mask[y3, x] < SELFIE_SEGMENTATION_CUTOFF_PROBABILITY:
                r3 = (x, y3)

    # calculate axes
    chest = scale * (abs(r1[0] - l1[0])/2)
    stomach = scale * (abs(r2[0] - l2[0])/2)
    waist = scale * (abs(r3[0] - l3[0])/2)

    return chest, stomach, waist


# Ramanujan approximation of ellipse perimeter
def ellipse_perimeter(major, minor):
    # major += 30
    # minor += 40
    # h = (major - minor)**2/(major + minor)**2
    # return np.pi * (major + minor) * (1 + (3 * h)/(10 + np.sqrt(4 - 3 * h)))

    return (major * 4) + (minor * 4)

# retrieve all lengths
def retreiveLengths(img_front, img_side, landmarks_front, landmarks_side, scale_factor):
    mm = {}

    # keypoints in world coordinates
    l_shoulder = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.LEFT_SHOULDER], scale_factor)
    r_shoulder = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.RIGHT_SHOULDER], scale_factor)
    l_wrist = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.LEFT_WRIST], scale_factor)
    r_wrist = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.RIGHT_WRIST], scale_factor)
    l_hip = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.LEFT_HIP], scale_factor)
    r_hip = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.RIGHT_HIP], scale_factor)
    l_ankle = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.LEFT_ANKLE], scale_factor)
    r_ankle = landmarkToWorldPoint(img_front, landmarks_front.landmark[mpHolistic.PoseLandmark.RIGHT_ANKLE], scale_factor)

    # cutlines for elliptical measurements
    hip_line_front = hipLine(img_front, landmarks_front)
    hip_line_side = hipLine(img_side, landmarks_side)
    shoulder_line_front = shoulderLine(img_front, landmarks_front)
    shoulder_line_side = shoulderLine(img_side, landmarks_side)

    # find ellipse axes for elliptical measurements
    majors = find_axes(img_front, hip_line_front, shoulder_line_front, scale_factor[0])
    minors = find_axes(img_side, hip_line_side, shoulder_line_side, scale_factor[0])

    # calculate straight lengths
    mm['shoulder'] = distance(l_shoulder, r_shoulder)
    mm['arms'] = np.mean( (distance(l_shoulder, l_wrist), distance(r_shoulder, r_wrist)) )
    mm['pantsLength'] = np.mean( (distance(l_hip, l_ankle), distance(r_hip, r_ankle)) )
    mm['shirtLength'] = np.mean( (distance(l_shoulder, l_hip), distance(r_shoulder, r_hip)) )

    # calculate elliptical perimeters
    mm['chest'] = ellipse_perimeter(majors[0], minors[0])
    mm['stomach'] = ellipse_perimeter(majors[1], minors[1])
    mm['waist'] = ellipse_perimeter(majors[2], minors[2])
 
    inches = {k: np.ceil(v/MM_PER_INCH) + HEURISTICS[k] for k, v in mm.items()}

    return inches

def magicMeasurements(calib_path, front_path, side_path):
    # open images
    img_calib, img_front, img_side = None, None, None
    try:
        img_calib = imutils.url_to_image(calib_path)
        img_front = imutils.url_to_image(front_path)
        img_side = imutils.url_to_image(side_path)
    except:
        return False, READ_ERROR

    # calculate commons
    scale = findScaleFactor(img_calib)
    landmarks_front = getPoseLandmarks(img_front)
    landmarks_side = getPoseLandmarks(img_side)

    if not scale or not landmarks_front or not landmarks_side:
        return False, PROCESS_ERROR

    scale_factor, error = scale

    # find lengths
    measurements = retreiveLengths(img_front, img_side, landmarks_front, landmarks_side, scale_factor)
    
    return True, measurements

if __name__ == "__main__":
    lengths = magicMeasurements(
        "samples/4/calib2.jpeg",
        "samples/4/front.jpeg",
        "samples/4/side.jpeg",
    )

    print(lengths)
