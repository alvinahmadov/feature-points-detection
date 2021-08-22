import cv2       # OpenCV

keypoints = []
descriptors = []

def process_frame(frame):
	# Convert the image to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Create BRISK algorithm
	# OpenCV default threshold = 30, octaves = 3
	# Using 4 octaves as cited as typical value by the original paper by Leutenegger et al.
	# Using 70 as detection threshold similar to real-world example of this paper
	brisk = cv2.BRISK_create(70, 4)
	# Combined call to let the BRISK implementation detect keypoints
	# as well as calculate the descriptors, based on the grayscale image.
	# These are returned in two arrays.
	(kps, descs) = brisk.detectAndCompute(gray, None)
	keypoints.append(kps)
	descriptors.append(descs)
	# 5. Use the generic drawKeypoints method from OpenCV to draw the 
	# calculated keypoints into the original image.
	# The flag for rich keypoints also draws circles to indicate
	# direction and scale of the keypoints.
	imgBrisk = cv2.drawKeypoints(gray, kps, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# 6. Finally, write the resulting image to the disk
	cv2.imshow('Frame', imgBrisk)
	pass



# 1. Load the original image
cap = cv2.VideoCapture('data/master.mp4')

if (cap.isOpened()== False):
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    # Display the resulting frame
    process_frame(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break


