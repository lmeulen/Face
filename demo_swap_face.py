import numpy as np
import cv2
import os
from face import FaceDetector

if __name__ == '__main__':

    # Read images
    img_new_face = cv2.imread(os.path.join('faces', 'ted_cruz.jpg'))
    img_original = cv2.imread(os.path.join('faces', 'hillary_clinton.jpg'))

    # Get faces
    detector = FaceDetector()
    new_face = detector.get_face(img_new_face)
    original_face = detector.get_face(img_original)

    # Swap face
    result = original_face.swap_face(img_original, img_new_face, new_face)

    # Combine original images and created image to single output and show
    result = np.concatenate((img_new_face, img_original, result), axis=1)
    result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Face Swapped", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
