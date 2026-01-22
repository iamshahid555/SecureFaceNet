import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class FaceClassifier:
    def __init__(self):
        """
        Simple classifier trained on face embeddings.
        """
        self.model = LogisticRegression(max_iter=1000)

    def train(self, embeddings, labels):
        """
        Train classifier on embeddings.
        """
        self.model.fit(embeddings, labels)

    def evaluate(self, embeddings, labels):
        """
        Evaluate classifier accuracy.
        """
        predictions = self.model.predict(embeddings)
        return accuracy_score(labels, predictions)

# if __name__ == "__main__":
#     from preprocessing.face_preprocessing import FacePreprocessor
#     from model.embedding_model import FaceEmbeddingModel

#     preprocessor = FacePreprocessor()
#     embedder = FaceEmbeddingModel()
#     classifier = FaceClassifier()

#     images, labels, names = preprocessor.load_lfw_data()

#     embeddings = []
#     valid_labels = []

#     for i in range(len(images)):
#         face = preprocessor.detect_and_preprocess(images[i])
#         if face is not None:
#             emb = embedder.get_embedding(face)
#             embeddings.append(emb)
#             valid_labels.append(labels[i])

#         if len(embeddings) >= 200:
#             break

#     embeddings = np.array(embeddings)
#     valid_labels = np.array(valid_labels)

#     # Simple train/test split
#     split = int(0.8 * len(embeddings))
#     X_train, X_test = embeddings[:split], embeddings[split:]
#     y_train, y_test = valid_labels[:split], valid_labels[split:]

#     classifier.train(X_train, y_train)
#     acc = classifier.evaluate(X_test, y_test)

#     print("Local client accuracy:", round(acc,2))