import numpy as np
from model.classifier import FaceClassifier


class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.classifier = FaceClassifier()

    def train(self, embeddings, labels):
        """
        Train locally and return model weights.
        """
        self.classifier.train(embeddings, labels)
        return self.classifier.model.coef_