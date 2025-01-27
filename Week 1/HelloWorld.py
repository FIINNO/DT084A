import cv2
import sys

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

def process_frame(frame):
	minVal = cv2.getTrackbarPos('minVal', 'Processed Frame') # Lower threshold
	maxVal = cv2.getTrackbarPos('maxVal', 'Processed Frame') # Upper threshold

	# Edge detection
	edges = cv2.Canny(frame, minVal, maxVal)
	return edges

def main():
	# Initialize the camera
	cap = initialize_camera()
	if cap is None:
		sys.exit(1)

	print("Camera feed started. Press 'q' to quit.")

	# Define named windows and trackbars
	cv2.namedWindow('Processed Frame')
	cv2.createTrackbar('minVal', 'Processed Frame', 0, 100, nothing)
	cv2.createTrackbar('maxVal', 'Processed Frame', 100, 400, nothing)


	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Check if frame is captured successfully
		if not ret:
			print("Error: Can't receive frame from camera.")
			break

		# Process the frame with a chosen (set) of functions
		output_frame = process_frame(frame)
        
		# Display the original frame
		cv2.imshow('Original Frame', frame)

		# Display the processed frame
		cv2.imshow('Processed Frame', output_frame)
        
		# Check for 'q' key press to quit the application
		# waitKey(1) returns -1 if no key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main() 