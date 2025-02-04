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


def process_frame(frame, method):
	if method == 1:
		minVal = cv2.getTrackbarPos('minVal', 'Processed Frame') # Lower threshold
		maxVal = cv2.getTrackbarPos('maxVal', 'Processed Frame') # Upper threshold
		frame = cv2.Canny(frame, minVal, maxVal)

	elif method == 2:
		cutoff_freq = cv2.getTrackbarPos('cutoff', 'Magnitude Spectrum')
		frame = high_pass_filter(frame, cutoff_freq)

	elif method == 3:
		frame = orb_feature_detector(frame)
		cv2.putText(frame,'ORB',(5,470), font, font_size, font_color, font_thickness, line_type)
		

	return frame

def update_windows(method):
	cv2.destroyWindow("Processed Frame")
	cv2.namedWindow("Processed Frame")
	cv2.namedWindow("Magnitude Spectrum")
	if method == 1:
		cv2.destroyWindow("Magnitude Spectrum")
		cv2.createTrackbar('minVal', 'Processed Frame', 0, 100, nothing)
		cv2.createTrackbar('maxVal', 'Processed Frame', 100, 400, nothing)
	if method == 2:
		cv2.namedWindow('Magnitude Spectrum')
		cv2.createTrackbar('cutoff', 'Magnitude Spectrum', 0, 100, nothing)
	if method == 3:
		cv2.destroyWindow("Magnitude Spectrum")


def menu(frame):

	cv2.putText(frame, "[1] Canny Edge Detection", (5, 15), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[2] High-pass Filter", (5, 30), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[3] Feature Detection", (5, 45), font, font_size, font_color, font_thickness, line_type)
	cv2.putText(frame, "[q] Exit", (5, 60), font, font_size, font_color, font_thickness, line_type)

	

def main():
	
	# Display menu

	cv2.namedWindow("Original Frame")
	cv2.namedWindow("Processed Frame")
	cv2.namedWindow("Magnitude Spectrum")

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
		cv2.imshow('Original Frame', frame)

		# Display the processed frame
		cv2.imshow('Processed Frame', output_frame)
        
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
		elif key == ord('q'):
			break

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main() 