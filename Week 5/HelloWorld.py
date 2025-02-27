import cv2
import sys
import numpy as np
from functions import *

# Global font values
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = (255, 255, 255)
font_thickness = 1
line_type = cv2.LINE_AA

image = None
images = []


# A function is required as argument in createTrackbar method
def nothing(x):
	pass


def initialize_camera(camera_index=0):
	cap = cv2.VideoCapture(camera_index)

	# Check if the cam is opened correctly
	if not cap.isOpened():
		print("Error: Could not open camera.")
		return None

	return cap

def process_images(image1, image2, nFeatures):
	kp1, desc1 = orb_feature_detector(image1, nFeatures)
	kp2, desc2 = orb_feature_detector(image2, nFeatures)
	matches = find_matches(desc1, desc2)

	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	# extracting positions of good matches
	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# homographic transformation + ransac filtering
	H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 4.0)

	# get inliers
	inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

    # draw inliers
	matched_img = np.zeros_like(image1)
	matched_img = cv2.drawMatches(
        image1, kp1, image2, kp2, inlier_matches, matched_img,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
	return matched_img, len(inlier_matches), H


def process_frame(frame, method):
	if method == 1:
		minVal = cv2.getTrackbarPos("minVal", "Processed Frame") # Lower threshold
		maxVal = cv2.getTrackbarPos("maxVal", "Processed Frame") # Upper threshold
		frame = cv2.Canny(frame, minVal, maxVal)

	elif method == 2:
		cutoff_freq = cv2.getTrackbarPos("cutoff", "Magnitude Spectrum")
		frame = high_pass_filter(frame, cutoff_freq)

	elif method == 3:
		nFeatures = cv2.getTrackbarPos("nFeatures", "Processed Frame") # number of features to be detected
		kp, desc = orb_feature_detector(frame, nFeatures)
		frame = cv2.drawKeypoints(frame, kp, None, color=(0, 0, 255), flags=0)

		cv2.putText(frame,"ORB",(5,470), font, font_size, font_color, font_thickness, line_type)

	elif method == 4:
		nFeatures = cv2.getTrackbarPos("nFeatures", "Processed Frame") # number of features to be detected
		kp, desc = orb_feature_detector(frame, nFeatures)
		frame = cv2.drawKeypoints(frame, kp, None, color=(0, 0, 255), flags=0)


		global images
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			images.append(frame.copy())
			if len(images) == 1:
				cv2.imshow("Captured Images", images[0])
			elif len(images) == 2:
				combined_images = cv2.hconcat([images[0], images[1]])
				cv2.imshow("Captured Images", combined_images)
		elif key == ord('m') and len(images) == 2:
			result, num_inliers, H = process_images(*images, nFeatures)
			cv2.imshow("Captured Images", result)
			images = []
	elif method == 5:
		global image
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			image = frame.copy()

		spatial_radius = cv2.getTrackbarPos("Spatial Radius", "Image Segmentation")
		color_radius = cv2.getTrackbarPos("Color Radius", "Image Segmentation")
		if not image is None:
			segmented_image = mean_shift_segmentation(image, spatial_radius, color_radius)
			cv2.imshow("Image Segmentation", segmented_image)
	elif method == 6:
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			images.append(frame.copy())
			if len(images) == 1:
				cv2.imshow("Captured Images", images[0])
			elif len(images) == 2:
				result_image = calculate_optical_flow(*images)
				combined_images = cv2.hconcat([images[0], result_image])
				cv2.imshow("Captured Images", combined_images)
			elif len(images) > 2:
				images = []


	return frame

def update_windows(method):
	cv2.destroyWindow("Processed Frame")
	cv2.namedWindow("Processed Frame")
	cv2.namedWindow("Magnitude Spectrum")
	cv2.namedWindow("Image Segmentation")
	if method == 1:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.destroyWindow("Image Segmentation")
		cv2.createTrackbar("minVal", "Processed Frame", 0, 100, nothing)
		cv2.createTrackbar("maxVal", "Processed Frame", 100, 400, nothing)
	elif method == 2:
		cv2.namedWindow("Magnitude Spectrum")
		cv2.destroyWindow("Image Segmentation")
		cv2.createTrackbar("cutoff", "Magnitude Spectrum", 0, 100, nothing)
	elif method == 3:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.destroyWindow("Image Segmentation")
		cv2.createTrackbar("nFeatures", "Processed Frame", 500, 1000, nothing)
	elif method == 4:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.destroyWindow("Image Segmentation")
		cv2.createTrackbar("nFeatures", "Processed Frame", 500, 1000, nothing)
	elif method == 5:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.destroyWindow("Processed Frame")
		cv2.createTrackbar("Spatial Radius", "Image Segmentation", 0, 100, nothing)
		cv2.createTrackbar("Color Radius", "Image Segmentation", 0, 100, nothing)
	elif method == 6:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.destroyWindow("Image Segmentation")


def menu(frame):

	cv2.putText(frame, "[1] Canny Edge Detection", (5, 15), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[2] High-pass Filter", (5, 30), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[3] Feature Detection", (5, 45), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[4] Feature Matching", (5, 60), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[5] Image Segmentation", (5, 75), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[6] Optical Flow", (5, 90), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[q] Exit", (5, 105), font, font_size, font_color, font_thickness, line_type)



def main():

	# Display menu

	cv2.namedWindow("Original Frame")
	cv2.namedWindow("Processed Frame")
	cv2.namedWindow("Magnitude Spectrum")
	cv2.namedWindow("Image Segmentation")

	selected_method = 1

	# Initialize the camera
	cap = initialize_camera()
	if cap is None:
		sys.exit(1)
	print("Camera feed started. Press 'q' to quit.")

	update_windows(selected_method)

	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Check if frame is captured successfully
		if not ret:
			print("Error: Can't receive frame from camera.")
			break

		# Process the frame with a chosen (set) of functions
		output_frame = process_frame(frame, selected_method)

		# Display the original frame
		menu(frame)
		cv2.imshow("Original Frame", frame)

		# Display the processed frame
		cv2.imshow("Processed Frame", output_frame)

		# Check for key press
		key = cv2.waitKey(1) & 0xFF
		if key == ord('1'):
			selected_method = 1
			update_windows(selected_method)
		elif key == ord('2'):
			selected_method = 2
			update_windows(selected_method)
		elif key == ord('3'):
			selected_method = 3
			update_windows(selected_method)
		elif key == ord('4'):
			selected_method = 4
			update_windows(selected_method)
		elif key == ord('5'):
			selected_method = 5
			update_windows(selected_method)
		elif key == ord('6'):
			selected_method = 6
			update_windows(selected_method)
		elif key == ord('q'):
			break

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
