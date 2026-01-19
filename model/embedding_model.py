import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1


class FaceEmbeddingModel:
    def __init__(self, device=None):
        """
        Initialize pretrained FaceNet model for embedding extraction.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def get_embedding(self, face):
        """
        Convert a preprocessed face image into a 512-dimensional embedding.
        """
        # Convert numpy array to torch tensor
        face_tensor = torch.tensor(face).unsqueeze(0).unsqueeze(0)

        # Convert grayscale to pseudo-RGB
        face_tensor = face_tensor.repeat(1, 3, 1, 1)
        face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.model(face_tensor)

        return embedding.cpu().numpy()[0]
    
if __name__ == "__main__":
    from preprocessing.face_preprocessing import FacePreprocessor

    preprocessor = FacePreprocessor()
    images, labels, names = preprocessor.load_lfw_data()

    for i in range(10):
        face = preprocessor.detect_and_preprocess(images[i])
        if face is not None:
            embedder = FaceEmbeddingModel()
            embedding = embedder.get_embedding(face)
            print("Embedding shape:", embedding.shape)
            print("Embedding sample:", embedding[:5])
            break