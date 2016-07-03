import cv2
import numpy as np

def Pause():
	while True:
		if cv2.waitKey(1) & 0xff == ord('q'):
			break


img = cv2.imread('img/car-plate1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Blur the image
blurred = cv2.GaussianBlur(img,(5,5),0)

# Find vertical edges. Note, derivative in X -> vert edges
# Derivative in y -> horizontal edges
# vertEdges = cv2.Sobel(blurred,-1,1,0) -> will only find one edge
# The other edge is negative, since datatype is uint8 -> pulled to zero -> cant see
# Use a signed type, take absolute, convert back to uint8 to see both edges
vertEdge64f = cv2.Sobel(blurred,cv2.CV_64F,1,0)
abs64f = np.absolute(vertEdge64f)
vertEdges = np.uint8(abs64f)

# Apply binary threshold to bring out the edges
# Use Otsu's method to determine optimal threshold value
retval,thre = cv2.threshold(vertEdges,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Get a structuring element -> for morphological operation
# i.e. dilation, erosion -> closing, opening
# To segment areas that might contain plates vs those that don't
structElem = cv2.getStructuringElement(cv2.MORPH_RECT,(50,20))
morphed = cv2.morphologyEx(thre,cv2.MORPH_CLOSE,structElem)

# Make a copy because findContours will modify source image
morphed_copy = morphed.copy()

# Find contours of the segments
# NOTE: If there are multiple return values, need to do it this way
# If single output var -> this var will contain the concat of the two variables, e.g. 
# one = (cnt,h)
cnt,h = cv2.findContours(morphed_copy,cv2.cv.CV_RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_NONE)
print 'Num of contours: {0}'.format(len(cnt))

# Get the top 20 largest contours
cnt_sorted = sorted(cnt,key=cv2.contourArea,reverse=True)[:2]

# For each contour, find the bounding rectangle of minimal area
# Using these rects, do some basic validation, e.g. check the aspect ratio, to filter off
# segments that are unlikely to contain car plates

# DEBUG: convert morphed to RGB else contours drawn will also be grayscale
morphed_color = cv2.cvtColor(morphed,cv2.COLOR_GRAY2BGR)

rectArray = []

for c in cnt_sorted:
	# Draw original contour
	cv2.drawContours(morphed_color,c,-1,(0,255,0),2)
	
	# draw bounding rect
	x,y,w,h = cv2.boundingRect(c) # returns (x,y,w,h)
	cv2.rectangle(morphed_color,(x,y),(x+w,y+h),(255,0,0),2)
	
	# draw minAreaRect
	rect = cv2.minAreaRect(c) # returns ((pt1.x, pt1.y),(pt2.x,pt2.y),degrees)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(morphed_color,[box],0,(0,0,255),2)

	cv2.imshow('morphed',morphed_color)
	Pause()
	cv2.destroyAllWindows()

# Crop these areas out
# resize to same width / height, and apply light histogram equalization -> better for training

cv2.imshow('morphed',morphed_color)
Pause()
