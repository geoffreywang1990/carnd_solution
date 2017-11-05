import os, glob, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CameraCalibrator:
    '''Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.'''
        
    def __init__(self, image_directory, image_filename, binary_filename, nx, ny):
        print ('Initializing CameraCalibrator ...')
        self.__image_directory = image_directory
        self.__image_filename = image_filename
        self.__binary_filename = binary_filename
        self.__nx = nx # the number of inside corners in x
        self.__ny = ny # the number of inside corners in y
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.__calibrated = False
        
    def __calibrate(self):
        
        # Read in and make a list of calibration images
        calibration_filenames = glob.glob(self.__image_directory+'/'+self.__image_filename)         
        
        # Arrays to store object points and image points from all the images
        object_points = [] # 3D points in real world space
        image_points = [] # 2D points in image plane
        
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...,(7,5,0)
        object_p = np.zeros((self.__ny*self.__nx,3),np.float32)
        object_p[:,:2] = np.mgrid[0:self.__nx,0:self.__ny].T.reshape(-1,2) # s,y coordinates
        
        # Extract the shape of any image
        image = cv2.imread(calibration_filenames[1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1] # (width,height)
        
        # Process each calibration image
        for image_filename in calibration_filenames:
            # Read in each image
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB is standard in matlibplot
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.__nx, self.__ny), None)
            
            # If found, draw corners
            if ret == True:
                # Store the corners found in the current image
                object_points.append(object_p) # how it should look like
                image_points.append(corners) # how it looks like

                # Draw and display the corners
                cv2.drawChessboardCorners(image, (self.__nx, self.__ny), corners, ret)
                plt.figure()
                plt.imshow(image)
                plt.show()
        
        # Do the calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, shape, None, None)
        #print(ret, mtx, dist, rvecs, tvecs)        
        
        # Pickle to save time for subsequent runs
        binary = {}
        binary["mtx"] = mtx
        binary["dist"] = dist
        binary["rvecs"] = rvecs
        binary["tvecs"] = tvecs
        pickle.dump(binary, open(self.__image_directory + '/' + self.__binary_filename, "wb"))
        
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.__calibrated = True
        
    def __load_binary(self):
        '''Load previously computed calibration binary data'''
        with open(self.__image_directory + '/' + self.__binary_filename, mode='rb') as f:
            binary = pickle.load(f)

        self.mtx = binary['mtx']
        self.dist = binary['dist']
        self.rvecs = binary['rvecs']
        self.tvecs = binary['tvecs']
        self.__calibrated = True
        
    def get_data(self):
        '''Getter for the calibration data. At the first call it gerenates it.'''
        if os.path.isfile(self.__image_directory + '/' + self.__binary_filename):
            self.__load_binary()
        else:
            self.__calibrate()
        return self.mtx, self.dist, self.rvecs, self.tvecs
    
    def undistort(self, image):
        if  self.__calibrated == False:
            self.get_data()
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
    def test_undistort(self, image_filename, plot=False):
        '''A method to test the undistort and to plot its result.'''
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB is standard in matlibplot
        image_undist = self.undistort(image)
        
        # Ploting both images Original and Undistorted
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original/Distorted')
        ax1.imshow(image)    
        ax2.set_title('Undistorted')
        ax2.imshow(image_undist)
        plt.show()
    