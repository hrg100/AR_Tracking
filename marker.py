import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# Based on tutorials from OpenCV
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

MIN_MATCH_COUNT = 10

class Marker(object):
    def __init__(self):
        self.query_key_points_ = None  # where points are
        self.query_descriptors_ = None  # describe those key points (region around as context);
        self.query_image_ = None  # input frame - usually from cam
        self.bounding_box_points = None
        self.detector = cv2.ORB_create(nfeatures=4000,nlevels=12)
        self.descriptor = cv2.ORB_create(nfeatures=4000,nlevels=12)

    def _apply_ratio_test(self, matches):
        # Need to draw only good matches, so create a mask
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance <= 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def _find_homography(self, matches, frame_key_points):
        # Need to draw only good matches, so create a mask
        # store all the good matches as per Lowe's ratio test.
        src_pts = np.float32([self.query_key_points_[m.queryIdx].pt for m in matches]).reshape(-1, 1,
                                                                                               2)  # cam good points
        dst_pts = np.float32([frame_key_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # train good points

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)  # ransac further elliminates outliers
        mask = mask.ravel().tolist()  # flatten mask to 1D from 2D
        return M, mask

    def query_marker(self, image_path):
        # Store image
        query_image_ = cv2.imread(image_path)
        query_image_ = cv2.resize(query_image_,(0,0),fx=0.2,fy=0.2)
        if query_image_ is None:
            return -1
        self.query_image_ = cv2.cvtColor(query_image_, cv2.COLOR_BGR2GRAY)


        self.query_key_points_ = self.detector.detect(self.query_image_)
        _, self.query_descriptors_ = self.descriptor.compute(self.query_image_, self.query_key_points_)
        self.query_descriptors_ = self.query_descriptors_.astype(np.float32)  # convert unsigned int 8bit
        return 1

    def find_marker(self, frame):
        # # # Convert frame to grayscale
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #         # # #
        #         # # # # Apply bilateral filtering
        #         # frame = cv2.bilateralFilter(frame, 5, 20, 20)
        #         # #
        #         # # # Apply canny edge detection
        #         # edge_map = cv2.Canny(frame, 40, 100)
        #         # cv2.imshow("Edge Map", edge_map)
        #         #
        #         # # # find contours in the edged image
        #         # contours, _ = cv2.findContours(edge_map.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        #         # black = np.zeros_like(edge_map)
        #         # cv2.drawContours(black, contours, -1, 255, 3)
        #         # cv2.imshow("Contours", black)

        # Detect SIFT keypoints and descriptors
        frame_key_points = self.detector.detect(frame)
        _, frame_descriptors = self.descriptor.compute(frame, frame_key_points)
        if len(frame_key_points) > 0:
            frame_descriptors = frame_descriptors.astype(np.float32)
        else:
            print("No keypoints found!")
            return -1, None, None
        kps_image = cv2.drawKeypoints(frame, frame_key_points, None, color=(255, 0, 0))
        cv2.imshow("Keypoints",kps_image)


        # Use FLANN Matcher: Fast Library for Approximate Nearest Neighbors algorithm
        # to get keypoint matches between query and train image
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # quality = 100
        flann = cv2.FlannBasedMatcher(index_params, search_params)  # set flann
        matches = flann.knnMatch(self.query_descriptors_, frame_descriptors,
                                 k=2)  # find matches based on nearest neighbors

        # Apply ratio test to filter out bad matches
        good_matches = self._apply_ratio_test(matches)

        if len(good_matches) > MIN_MATCH_COUNT:

            # Find homography given good matches
            M, mask = self._find_homography(good_matches, frame_key_points)

            if M is None:
                print("Unable to estimate homography!")
                return -1, None, None

            h, w = self.query_image_.shape
            # four corners of query image
            # BL, BR, TR, TL
            pts = np.float32([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]]).reshape(-1, 1,
                                                                                       2)  # don't know how many rows, Mx1x2. pts are boundaries of query img

            # warp query image to coordinate system of train image
            self.bounding_box_points = cv2.perspectiveTransform(pts, M)

            ret, IM = cv2.invert(M)
            warped_frame = cv2.warpPerspective(frame,IM,(self.query_image_.shape[1],self.query_image_.shape[0]))
            cv2.imshow("Wapred Query Image",warped_frame)

            outlined_train_image = cv2.polylines(frame, [np.int32(self.bounding_box_points)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=mask,  # draw only inliers
                               flags=2)

            matches_image = cv2.drawMatches(self.query_image_, self.query_key_points_, outlined_train_image,
                                            frame_key_points, good_matches, None, **draw_params)
            matches_image = imutils.resize(matches_image, height=500)

            print("Good maches: {}".format(len(good_matches)))
            return 1, outlined_train_image, matches_image
        # plt.imshow(outlined_train_image, 'gray')
        # plt.show()

        else:
            print("Only {} good matches found; not enough!".format(len(good_matches)))
            return -1, None, None

    def estimate_pose(self, frame, camera):

        ##### CAMERA CALIBRATION USING HEURISTICS
        # Frame width and height
        # h, w = frame.shape[:2]
        # Focal Length
        # f = w
        # Optical center
        # o_x = 0.5 * w
        # o_y = 0.5 * h
        # Camera matrix
        # camera_matrix = np.float32([[f, 0, 0.5 * o_x],
        #                 [0, f, 0.5 * o_y],
        #                 [0.0, 0.0, 1.0]])
        # Assume no radial distortion
        # dist_coeffs = np.zeros(4)

        ##### ACTUAL CAMERA CALIBRATION
        #camera_matrix = np.float32([[997.380354794108, 0, 0], [0, 992.053293807353, 0], [516.511429768328, 399.653733578975, 1]])
        camera_matrix = np.float32(
            [[997.380354794108, 0, 0], [0, 992.053293807353, 0], [516.511429768328, 399.653733578975, 1]])
        #dist_coeffs = np.float64([-0.008, 0.179, 0.000, 0.000])
        dist_coeffs = np.zeros(4)

        # Setup 3D model coordinates
        h, w = self.query_image_.shape[:2]
        max_dim = max(h, w)
        unitW = w / 2
        unitH = h / 2

        # BL, BR, TR, TL
        object_3d_coordinates = [(-unitW, unitH, 0), (unitW, unitH, 0), (unitW, -unitH, 0), (-unitW, -unitH, 0)]

        # Frame height and width
        h, w = frame.shape[:2]

        # Build array of 3d object coordinates; set Z=0
        c_x = 0
        c_y = 0

        for p in self.bounding_box_points:
            x, y = p[0]  # pull out coords
            c_x += x
            c_y += y

        c_x = c_x//4
        c_y = c_y//4

        object_3d_coordinates = np.float32(object_3d_coordinates)
        image_2d_coordinates = self.bounding_box_points

        # Determine rvec and tvec
        ret, rvec, tvec = cv2.solvePnP(object_3d_coordinates, image_2d_coordinates, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)  # fing obj pose fron 2d 3d point correspondences
        if ret:
            print(rvec)
            print(tvec)
            # R = cv2.Rodrigues(rvec)[0]
            # R = R.T
            # tvec = np.matmul(-R,tvec.flatten())
            # rvec = cv2.Rodrigues(R)[0]
            # tvec = tvec.reshape(-1,1)

            # Backproject points
            tl, _ = cv2.projectPoints(object_3d_coordinates[0], rvec, tvec, camera_matrix, dist_coeffs)
            tr, _ = cv2.projectPoints(object_3d_coordinates[1], rvec, tvec, camera_matrix, dist_coeffs)
            br, _ = cv2.projectPoints(object_3d_coordinates[2], rvec, tvec, camera_matrix, dist_coeffs)
            bl, _ = cv2.projectPoints(object_3d_coordinates[3], rvec, tvec, camera_matrix, dist_coeffs)
            cv2.circle(frame, (tl[0][0][0], tl[0][0][1]), 10, (255, 0, 255), -1)
            cv2.circle(frame, (tr[0][0][0], tr[0][0][1]), 10, (255, 0, 255), -1)
            cv2.circle(frame, (br[0][0][0], br[0][0][1]), 10, (255, 0, 255), -1)
            cv2.circle(frame, (bl[0][0][0], bl[0][0][1]), 10, (255, 0, 255), -1)

            # Create axis for displaying orientation
            z_axis = np.array([(0.0, 0.0, 1000.0)])
            y_axis = np.array([(0.0, 1000.0, 0.0)])
            x_axis = np.array([(1000.0, 0.0, 0.0)])

            # Project axis to image plane
            (z_axis_projected, _) = cv2.projectPoints(z_axis, rvec, tvec, camera_matrix, dist_coeffs)
            (y_axis_projected, _) = cv2.projectPoints(y_axis, rvec, tvec, camera_matrix, dist_coeffs)
            (x_axis_projected, _) = cv2.projectPoints(x_axis, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw line from center of image to projected point
            origin = (int(c_x), int(c_y))
            z_proj_pt = (int(z_axis_projected[0][0][0]), int(z_axis_projected[0][0][1]))
            y_proj_pt = (int(y_axis_projected[0][0][0]), int(y_axis_projected[0][0][1]))
            x_proj_pt = (int(x_axis_projected[0][0][0]), int(x_axis_projected[0][0][1]))

            cv2.line(frame, z_proj_pt, origin, (150, 0, 0), 2)
            cv2.line(frame, y_proj_pt, origin,  (0, 150, 0), 2)
            cv2.line(frame, x_proj_pt, origin,  (0, 0, 150), 2)

        return 1, frame
