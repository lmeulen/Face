# Import required modules
import cv2
from face import FaceDetector
import utils.utils as utls

WEBCAM = False


def process_frame(image, detector, overlay=True):
    original = image.copy()
    faces = detector.detect_faces(image)
    edit_image = None
    for f in faces:
        if overlay:
            edit_image = original
        else:
            height, width = image.shape[:2]
            edit_image = utls.create_blank_image(width, height)

        # Add landmarks to image, cover eyes and draw triangles
        # f.draw_landmarks(edit_image, dots=True, lines=True, cog=True, face_ellipse=True, eyes=True, cog_lines=True)
        f.draw_landmarks(edit_image, face_ellipse=True)
        # f.cover_eyes(edit_image)
        #f.draw_delaunay(edit_image)

        # Extract and show image
        # face_image = f.extract_aligned_face(original)
        # cv2.imshow("face", face_image)

        # Show exploded view
        # exploded_view = f.exploded_view_delaunay(edit_image)
        # cv2.imshow("Exploded delaunay", exploded_view)
        # f.normalize_triangles()

        # Blur face
        blurred_face= f.blur_face(edit_image)
        cv2.imshow("Blurred face", blurred_face)
        f.normalize_triangles()

    # Display the edit_image
    cv2.imshow("image", edit_image)

        
if __name__ == "__main__":    
    det = FaceDetector()
    if WEBCAM:
        # Set up webcam
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            process_frame(frame, det)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        frame = cv2.imread("obama.jpg")
        process_frame(frame, det, overlay=True)
        cv2.waitKey(0)
