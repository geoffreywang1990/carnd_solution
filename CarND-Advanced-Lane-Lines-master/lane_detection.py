import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt

class LaneDetector:
    
    def __init__(self):
        print ('Initializing LaneDetector ...')
        
    def compute_binary_image(self, color_image, plot=False):    
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(color_image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        
        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        if (plot):
            # Ploting both images Original and Binary
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Undistorted/Color')
            ax1.imshow(color_image)    
            ax2.set_title('Binary/Combined S channel and gradient thresholds')
            ax2.imshow(combined_binary, cmap='gray')
            plt.show()
        
        return combined_binary
    
    def compute_perspective_transform(self, binary_image):
        # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
        shape = binary_image.shape[::-1] # (width,height)
        w = shape[0]
        h = shape[1]
        transform_src = np.float32([ [580,450], [160,h], [1150,h], [740,450]])
        transform_dst = np.float32([ [0,0], [0,h], [w,h], [w,0]])
        M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        return M
    
    def apply_perspective_transform(self, binary_image, M, plot=False):
        warped_image = cv2.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
        if(plot):
            # Ploting both images Binary and Warped
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Binary/Undistorted and Tresholded')
            ax1.imshow(binary_image, cmap='gray')
            ax2.set_title('Binary/Undistorted and Warped Image')
            ax2.imshow(warped_image, cmap='gray')
            plt.show()

        return warped_image

    def extract_lanes_pixels(self, binary_warped):
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        #plt.plot(histogram)
        #plt.show()
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds
            
    def poly_fit(self, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, binary_warped, plot:False):  
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        if(plot):
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
        
        return left_fit, right_fit, ploty, left_fitx, right_fitx 
        
        
    def compute_curvature(self, left_fit, right_fit, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty):
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        y_eval = np.max(ploty)
    
        fit_cr_left = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        curverad_left = ((1 + (2 * left_fit[0] * y_eval / 2. + fit_cr_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_left[0])
        fit_cr_right = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        curverad_right = ((1 + (2 * left_fit[0] * y_eval / 2. + fit_cr_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_right[0])
    
        return (curverad_left + curverad_right) / 2
    
    def pain_lane(self, undist, warped, M, left_fitx, right_fitx, ploty, plot=False):
        
        Minv = inv (M)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        if(plot):
            plt.imshow(result)
            plt.show()
        
        return result
    
    def compute_center_offset(self, curverad, undist_image, plot=False):
        
        xm_per_pix = 3.7/700
        lane_center_x = int(curverad)
        image_center_x = int(undist_image.shape[1] / 2)
        offset_from_centre = (image_center_x - lane_center_x) * xm_per_pix # in meters
        
        return offset_from_centre
        
        
    def render_curvature_and_offset(self, undist_image, curverad, offset, plot=False):   
        
        # Add curvature and offset information
        offst_text = 'offset: {:.2f}m'.format(offset)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undist_image, offst_text, (24, 50), font, 1, (255, 255, 255), 2)

        curverad_text = 'curverad: {:.2f}m'.format(curverad)
        cv2.putText(undist_image, curverad_text, (19, 90), font, 1, (255, 255, 255), 2) 
        
        if(plot):
            plt.imshow(undist_image)
            plt.show()
        
        return undist_image
    