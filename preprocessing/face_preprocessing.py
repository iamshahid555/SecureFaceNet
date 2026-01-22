import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people


class FacePreprocessor:
    def __init__(self, image_size=(160, 160)):
        """
        image_size: target size for face images (FaceNet standard is 160x160)
        """
        self.image_size = image_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_lfw_data(self, min_faces_per_person=20):
        """
        Load LFW dataset using scikit-learn.
        Images are returned as grayscale numpy arrays.
        """
        lfw = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=0.5
        )
        return lfw.images, lfw.target, lfw.target_names

    def detect_and_preprocess(self, image):
        """
        Detect face, crop, resize, and normalize.
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)

        faces = self.face_cascade.detectMultiScale(
            image_uint8,
            scaleFactor=1.1,
            minNeighbors=5
        )

        if len(faces) == 0:
            return None

        # Take the first detected face
        x, y, w, h = faces[0]
        face = image_uint8[y:y+h, x:x+w]

        # Resize to FaceNet input size
        face = cv2.resize(face, self.image_size)

        # Normalize to [0, 1]
        face = face.astype(np.float32) / 255.0

        return face
 
# if __name__ == "__main__":
#     preprocessor = FacePreprocessor()
#     images, labels, names = preprocessor.load_lfw_data()

#     print("Testing face detection on multiple images...")

#     for i in range(10):
#         face = preprocessor.detect_and_preprocess(images[i])
#         if face is not None:
#             print(f"Face detected at index {i}")
#             print("Face shape:", face.shape)
#             print("Pixel range:", face.min(), face.max())
#             break
#     else:
#         print("No face detected in first 10 images")