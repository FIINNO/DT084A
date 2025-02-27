import cv2
import numpy as np

def high_pass_filter(frame, cutoff_freq):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale

	f = np.fft.fft2(frame) # fourier transform to move into frequency domain
	fshift = np.fft.fftshift(f) # shifting the zero freq component to the center

	# create filter mask
	rows, cols, = frame.shape
	cx, cy = cols // 2, rows // 2
	mask = np.ones((rows, cols), np.uint8)
	cv2.circle(mask, (cx, cy), cutoff_freq, 0, -1) # remove low frequencies


	fshift_filtered = fshift * mask # applying filter

	# magnitude spectrum visualization
	magnitude_spectrum = 20 * np.log(np.abs(fshift_filtered) + 1)
	cv2.imshow("Magnitude Spectrum", magnitude_spectrum.astype(np.uint8))

	# transform back to spatial domain
	f_ishift = np.fft.ifftshift(fshift_filtered)
	frame_filtered = np.fft.ifft2(f_ishift)
	frame_filtered = np.abs(frame_filtered)

	# normalize
	frame_filtered = cv2.normalize(frame_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return frame_filtered


def harris_feature_detector(frame, threshold=0.3, alpha=0.04):

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    I_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Auto-correlation matrix/structure tensor components
    I_x2 = I_x**2
    I_y2 = I_y**2
    I_xy = I_x*I_y

    # Apply gaussian smoothing to reduce noise
    I_x2 = cv2.GaussianBlur(I_x2, (3, 3), 1)
    I_y2 = cv2.GaussianBlur(I_y2, (3, 3), 1)
    I_xy = cv2.GaussianBlur(I_xy, (3, 3), 1)

    # Compute corner response function
    det_A = I_x2*I_y2 - (I_xy**2)
    trace_A = I_x2 + I_y2
    R = det_A - alpha*(trace_A**2)

    # Determine corners based on threshold
    R_norm = cv2.normalize(R, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    feature_map = R_norm > threshold

    # Convert detected corners to cv2.KeyPoint objects
    kp = [cv2.KeyPoint(float(x), float(y), 1) for y, x in np.argwhere(feature_map)]

    img_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 0 , 255), flags = 0)
    return img_kp

def orb_feature_detector(frame, nFeatures=500):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nFeatures)

    kp = orb.detect(img_gray, None)
    kp, desc = orb.compute(img_gray, kp)

    return kp, desc

def find_matches(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

def apply_homography(image1, image2, H):
    height, width = image2.shape[:2]
    transformed_image = cv2.warpPerspective(image1, H, (width, height))
    return transformed_image


def mean_shift_segmentation(image, spatial_radius=20, color_radius=40):
    segmented_image = cv2.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=color_radius)
    return segmented_image

def calculate_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp1, desc = orb_feature_detector(frame1)

    # convert keypoints to numpy array (needed for LK optical flow)
    p0 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)

    # lucas kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    output = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(output)
    # draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        output = cv2.circle(output, (int(a), int(b)), 3, (0, 0, 255), -1)

    final_output = cv2.add(output, mask)
    return final_output
