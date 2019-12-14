import sys
from marker import Marker
from camera import Camera
import cv2

'''
References:

https://medium.com/@ahmetozlu93/marker-less-augmented-reality-by-opencv-and-opengl-531b2af0a130
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

'''

def main():
	# Parse command line
	if len(sys.argv) < 2:
		print("python main.py mode <query_image_path> [ <train_video_path> ]")
		return 1

	# Extract features and descriptors from query image
	query_image_path = sys.argv[1]
	print("Query Image Path: {}".format(query_image_path))
	marker = Marker() #class maker
	ret = marker.query_marker(query_image_path) #method from class; query is scan of card
	if ret == -1:
		print("Unable to get query image...terminating...")
		return

	cap = None
	if len(sys.argv) == 3:
		train_image_path = sys.argv[2] #live scene
		print("Using video file...Training Image Path: {}".format(train_image_path))
		cap = cv2.VideoCapture(train_image_path)
	else:
		print("Using web cam")
		cap = cv2.VideoCapture(0)

	# For loading and saving videos
	id_ = 3
	start_frame = 0
	# Get metadata from input movie
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	# Create video writer
	video_writer = cv2.VideoWriter('videos/output_{}.avi'.format(id_), fourcc, fps,
									(frame_width, frame_height))

	i = 0
	while True:
		ret, frame = cap.read()
		i+=1
		# Display every 5 frames
		if i % 5 == 0:
			continue
		if ret == False:
			break
		if ret == -1:
			print("Unable to get frame...terminating...")
			return
		ret, outlined_frame, matches_image = marker.find_marker(frame) #drawn on webcam frame (blue outline)
		if ret != -1: #if no error
			frame = outlined_frame
			marker.estimate_pose(outlined_frame,Camera()) #get pose from drawn image
			cv2.imshow('Matches',matches_image)

		cv2.imshow('AR Frame',frame)
		key = cv2.waitKey(1)
		print(key)
		if key & 0xFF == ord('q'):
			break
		video_writer.write(frame)

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
