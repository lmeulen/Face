# Import required modules
import cv2
import dlib
from collections import OrderedDict
import numpy as np
import utils.utils as utls


class Face:
    # For dlib’s 68-point facial landmark detector:
    LANDMARK_68_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("inner_mouth", (60, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])
    LANDMARK_MOST_LEFT = 0
    LANDMARK_MOST_RIGHT = 16

    landmarks = None
    center_of_gravity = None
    left_center = None
    right_center = None
    center_eyes = None
    ellipse = None
    triangles = None

    def __init__(self, landmarks=None):
        self.set_landmarks(landmarks)

    def set_landmarks(self, landmarks):
        """
        Set new landmarks, landmarks are stored as a numpy array
        :landmarks array of landmarks
        :image image containing the face
        """
        self.landmarks = self.shape_to_np(landmarks)
        self.center_of_gravity = None
        self.left_center = None
        self.right_center = None
        self.center_eyes = None
        self.ellipse = None
        self.triangles = None

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        """
        Convert a landmark shape object to a numpy array of coordinates
        :param shape: original shape
        :param dtype: datatype of the resulting coordinatis (default int)
        :return: numpy array ofcoordinates (x,y)
        """
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def get_center_of_gravity(self):
        """
        Get the center of gravity
        :return center of gravity (x,y)
        """
        if self.center_of_gravity is None:
            xlist = self.landmarks[:, 0]
            ylist = self.landmarks[:, 1]
            self.center_of_gravity = (int(np.mean(xlist)), int(np.mean(ylist)))
        return self.center_of_gravity

    def cover_eyes(self, image):
        """
        Cover the eyes with a black bar. Rectangle covers
        the eyes from the top of eye brows till the middle
        of the nose. Works regardles head angle.
        :param image: image containing the face
        """
        # left_Eyebrow + roght_eyebrow + left_Eye + right_eye + 0 + 17
        idxs = np.r_[0, 16, 17:26, 29, 36:47]
        box_area = self.landmarks[idxs]
        rect = cv2.minAreaRect(box_area)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 0), -1)

    def get_eye_centers(self):
        """
        Extract the eye centers from the landmarks
        :return left_center, right_center, middle_of_eyes
        """
        if self.left_center is None:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = self.LANDMARK_68_IDXS["left_eye"]
            (rStart, rEnd) = self.LANDMARK_68_IDXS["right_eye"]
            left_eye = self.landmarks[lStart:lEnd]
            right_eye = self.landmarks[rStart:rEnd]
            # compute the center for the eyes
            self.left_center = left_eye.mean(axis=0).astype("int")
            self.right_center = right_eye.mean(axis=0).astype("int")
            # compute center
            self.center_eyes = ((self.left_center[0] + self.right_center[0]) // 2,
                                (self.left_center[1] + self.right_center[1]) // 2)
        return self.left_center, self.right_center, self.center_eyes

    def get_face_ellipse(self):
        """
        Determine ellipse based on jawline
        :return ellipse parameters [cx,cy,a,b,angle]
        """
        if self.ellipse is None:
            (start, end) = self.LANDMARK_68_IDXS["jaw"]
            jawline = self.landmarks[start:end]
            ellipse = utls.fit_ellipse(jawline)
            self.ellipse = list(map(int, ellipse))
            # Surrounding ellipse  [cx,cy,a,b,angle]
            return self.ellipse

    def draw_landmarks(self, image, line_color=(0, 255, 0), dots_color=(0, 255, 0), dots=False,
                       lines=False, cog=False, face_ellipse=False, eyes=False, cog_lines=False, copy=False):

        frame = image.copy() if copy else image
        landmarks = self.landmarks
        if dots:
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, dots_color, thickness=2)

        if face_ellipse:
            e = self.get_face_ellipse()
            cv2.ellipse(frame, (e[0], e[1]), (e[2], e[3]), e[4], 0, 360, (0, 255, 0), 1)

        if eyes:
            lec, rec, cen = self.get_eye_centers()
            cv2.line(frame, (lec[0], lec[1]), (rec[0], rec[1]), (0, 255, 0), thickness=1)
            cv2.circle(frame, cen, 2, (255, 0, 0), thickness=4)

        if cog_lines:
            cog = self.get_center_of_gravity()
            for mark in landmarks:
                cv2.line(frame, (mark[0], mark[1]), cog, (0, 255, 0), thickness=1)

        if lines:
            # Around Chin. Ear to Ear
            for i in range(1, 17):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)

                # Line on top of nose
            for i in range(28, 31):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)

                # left eyebrow
            for i in range(18, 22):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
                # right eyebrow
            for i in range(23, 27):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
                # Bottom part of the nose
            for i in range(31, 36):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
                # Line from the nose to the bottom part above
            cv2.line(frame, (landmarks[30][0], landmarks[30][1]), (landmarks[35][0], landmarks[35][1]), line_color,
                     thickness=1)

            # Left eye
            for i in range(37, 42):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
            cv2.line(frame, (landmarks[36][0], landmarks[36][1]), (landmarks[41][0], landmarks[41][1]), line_color,
                     thickness=1)

            # Right eye
            for i in range(43, 48):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
            cv2.line(frame, (landmarks[42][0], landmarks[42][1]), (landmarks[47][0], landmarks[47][1]), line_color,
                     thickness=1)

            # Lips outer part
            for i in range(49, 60):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
            cv2.line(frame, (landmarks[48][0], landmarks[48][1]), (landmarks[59][0], landmarks[59][1]), line_color,
                     thickness=1)

            # Lips inside part
            for i in range(61, 68):
                cv2.line(frame, (landmarks[i][0], landmarks[i][1]), (landmarks[i - 1][0], landmarks[i - 1][1]),
                         line_color, thickness=1)
            cv2.line(frame, (landmarks[60][0], landmarks[60][1]), (landmarks[67][0], landmarks[67][1]), line_color,
                     thickness=1)

        if cog:
            cv2.circle(frame, self.get_center_of_gravity(), 2, (255, 0, 0), thickness=2)

        return frame

    def extract_aligned_face(self, image, size=256, target_left_eye=(0.35, 0.35)):
        """
        Extract the face described by the landmarks. Align face during extract.
        :param image: original image
        :param size: size of the extracted image (square)
        :param target_left_eye: target location of the left eye
        :size target size of resulting image height == width, default = 256
        :target_left_eye relative coordinate to place left eye, default (0.35, 0.35)
        :return image with aligned face of specified size
        """

        # Find eyes
        left_center, right_center, center_eyes = self.get_eye_centers()

        # compute the angle and distance between the eye centers
        deltax = right_center[0] - left_center[0]
        deltay = right_center[1] - left_center[1]
        angle = np.degrees(np.arctan2(deltay, deltax)) - 180
        distance = np.sqrt((deltax ** 2) + (deltay ** 2))

        # calculate scale based upon eye distance in the X-plance
        target_distance = (1.0 - 2 * target_left_eye[0]) * size
        scale = target_distance / distance

        # create a rotation matrix for rotating and scaling the face
        rotmatrix = cv2.getRotationMatrix2D(center_eyes, angle, scale)

        # update the translation component of the matrix
        t_x = size * 0.5
        t_y = size * target_left_eye[1]
        rotmatrix[0, 2] += (t_x - center_eyes[0])
        rotmatrix[1, 2] += (t_y - center_eyes[1])

        # apply the affine transformation
        face = cv2.warpAffine(image, rotmatrix, (size, size), flags=cv2.INTER_CUBIC)
        # return the aligned face
        return face

    def calculate_delaunay_triangles(self, image):
        """
        Calculate the delauny triangles
        :return list of triangles ((x1,y1), (x2,y2), (x3,y3))).
        """
        triangles = []
        if self.triangles is None:
            height, width = image.shape[:2]
            rect = (0, 0, width, height)
            # Create subdiv
            subdiv = cv2.Subdiv2D(rect)
            # Insert points into subdiv
            points = self.landmarks
            for p in points:
                subdiv.insert((p[0], p[1]))
            # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
            trianglelist = subdiv.getTriangleList()
            # Find the indices of triangles in the points array
            for t in trianglelist:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                if utls.rect_contains(rect, pt1) and utls.rect_contains(rect, pt2) and utls.rect_contains(rect, pt3):
                    triangles.append((pt1, pt2, pt3))
            self.triangles = triangles
        return triangles

    def draw_delaunay(self, image, color=(0, 255, 0)):

        trianglelist = self.calculate_delaunay_triangles(image)
        size = image.shape
        r = (0, 0, size[1], size[0])

        for t in trianglelist:
            pt1 = t[0]
            pt2 = t[1]
            pt3 = t[2]
            if utls.rect_contains(r, pt1) and utls.rect_contains(r, pt2) and utls.rect_contains(r, pt3):
                cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA, 0)
                cv2.line(image, pt2, pt3, color, 1, cv2.LINE_AA, 0)
                cv2.line(image, pt3, pt1, color, 1, cv2.LINE_AA, 0)

    @staticmethod
    def move_triangle(triangle, displacement):
        """
        Move a triangle in x and y direction
        :param triangle: Triangle to move [p1,p2,p3]
        :param displacement: Displacement in 2 dimensions (dx, dy)
        :return moved triangle
        """
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]
        (dx, dy) = displacement
        p1 = (int(p1[0] + dx), int(p1[1] + dy))
        p2 = (int(p2[0] + dx), int(p2[1] + dy))
        p3 = (int(p3[0] + dx), int(p3[1] + dy))
        return p1, p2, p3

    def normalize_triangles(self, size=250):
        """
        Normalize triangle so they are located top left
        :return normalized triangles
        """
        min_width = min_height = 1000
        max_width = max_height = 0
        for tr in self.triangles:
            min_width = min(min_width, tr[0][0], tr[1][0], tr[2][0])
            max_width = max(max_width, tr[0][0], tr[1][0], tr[2][0])
            min_height = min(min_height, tr[0][0], tr[1][0], tr[2][0])
            max_height = max(max_height, tr[0][0], tr[1][0], tr[2][0])
        factor = size / max((max_width - min_width), (max_height - min_height))
        norm_tr = []
        for tr in self.triangles:
            point1 = [int((tr[0][0] - min_width) * factor), int((tr[0][1] - min_height) * factor)]
            point2 = [int((tr[1][0] - min_width) * factor), int((tr[1][1] - min_height) * factor)]
            point3 = [int((tr[2][0] - min_width) * factor), int((tr[2][1] - min_height) * factor)]
            norm_tr.append((point1, point2, point3))
        return norm_tr

    def exploded_view_delaunay(self, image, size=250, factor=1, color=(0, 255, 0)):
        h = w = size
        height = h + int((factor * h)) + 1
        width = w + int((factor * w)) + 1
        frame = utls.create_blank_image(width, height)
        r = (0, 0, width, height)

        self.calculate_delaunay_triangles(image)
        trianglelist = self.normalize_triangles()
        height, width = frame.shape[:2]
        center = (width / 2, height / 2)
        for t in trianglelist:
            dx = 0
            dy = 0
            if not utls.point_inside_triangle(center, t):
                cot = utls.center_of_triangle(t)
                dx = int(center[0] + ((cot[0] - center[0]) * factor))
                dy = int(center[1] + ((cot[1] - center[1]) * factor))
            t2 = self.move_triangle(t, (dx, dy))
            pt1 = t2[0]
            pt2 = t2[1]
            pt3 = t2[2]
            if utls.rect_contains(r, pt1) and utls.rect_contains(r, pt2) and utls.rect_contains(r, pt3):
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA, 0)
                cv2.line(frame, pt2, pt3, color, 1, cv2.LINE_AA, 0)
                cv2.line(frame, pt3, pt1, color, 1, cv2.LINE_AA, 0)
        return frame


class FaceDetector:
    # Shape predictor model from DLIB
    # source: https://github.com/davisking/dlib-models
    SHAPE_MODEL = "models\shape_predictor_68_face_landmarks.dat"

    detector = None
    predictor = None

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.SHAPE_MODEL)

    @staticmethod
    def equalize_image(image):
        """
        Apply Global Histogram Equalization
        In order to increase cotrast

        :param image: original image
        :return normalized image (grayscale)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return histogram.apply(gray)

    def detect_faces(self, image_org):
        image = image_org.copy()
        image = self.equalize_image(image)
        # Detect the faces in the image
        detected_faces = self.detector(image, 1)
        faces = []
        for k, face in enumerate(detected_faces):
            # Find landmarks
            landmarks = self.predictor(image, face)
            f = Face(landmarks)
            faces.append(f)
        return faces
