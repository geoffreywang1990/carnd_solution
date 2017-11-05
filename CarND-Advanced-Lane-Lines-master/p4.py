'''
The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

import numpy as np
import glob
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera_calibration import CameraCalibrator
from lane_detection import LaneDetector

# Calibration variables
CALIBRATION_DIRECTORY = "./camera_cal/"
CALIBRATION_FILENAME = "calibration*.jpg"
CALIBRATION_BINARY_FILENAME = "calibration.p"
CALIBRATION_NX = 9
CALIBRATION_NY = 6

# Test image variables
TEST_DIRECTORY = './test_images/'
TEST_FILENAME = "*.jpg"
OUTPUT_DIRECTORY = './output_images'


def process_image(image, plot=False):
    
    # Apply a distortion correction to raw images.
    undistorted_image = calibrator.undistort(image)
    
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # Compute the binary image of the undistorted image
    binary_image = lane_detector.compute_binary_image(undistorted_image, plot)
        
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    # Compute the transform image of the binary image
    M = lane_detector.compute_perspective_transform(binary_image)
    binary_warped = lane_detector.apply_perspective_transform(binary_image, M, plot) 
    
    # Detect lane pixels and fit to find the lane boundary.  
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = lane_detector.extract_lanes_pixels(binary_warped)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    left_fit, right_fit, ploty, left_fitx, right_fitx = lane_detector.poly_fit(leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, binary_warped, plot)
    curverad = lane_detector.compute_curvature(left_fit, right_fit, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty)

    # Warp the detected lane boundaries back onto the original image.
    new_image = lane_detector.pain_lane(undistorted_image, binary_warped, M, left_fitx, right_fitx, ploty, plot)
    
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    offset_from_centre = lane_detector.compute_center_offset(curverad, new_image, plot)
    
    new_image = lane_detector.render_curvature_and_offset(new_image, curverad, offset_from_centre, plot)
    
    return new_image


def process_test_images(lane_detector, plot=False):
    
    test_filenames = glob.glob(TEST_DIRECTORY+'/'+TEST_FILENAME) 
    # Process each test image
    for image_filename in test_filenames:
        # Read in each image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB is standard in matlibplot
        
        image = process_image(image, plot)
        
        cv2.imwrite(OUTPUT_DIRECTORY+'/'+image_filename.split('/')[-1], image);
  
def process_video(video_filename, lane_detector, plot=False):
    video_input = VideoFileClip(video_filename + ".mp4")
    video_output = video_input.fl_image(process_image)
    video_output.write_videofile(video_filename + "_output.mp4", audio=False)

 
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Initialize the camera calibrator
calibrator = CameraCalibrator(CALIBRATION_DIRECTORY, CALIBRATION_FILENAME, CALIBRATION_BINARY_FILENAME, CALIBRATION_NX, CALIBRATION_NY)
# Test one image
calibrator.test_undistort(CALIBRATION_DIRECTORY+'/calibration5.jpg', plot=True)


# It implements methods related to the lane processing
lane_detector = LaneDetector()

# Process the test images and the video
process_test_images(lane_detector, plot=True)
process_video("project_video", lane_detector, plot=False)
    
