import numpy as np
from federated.fedavg import federated_average


class FederatedServer:
    def __init__(self):
        self.global_weights = None

    def aggregate(self, client_weights):
        """
        Aggregate client weights using FedAvg.
        """
        self.global_weights = federated_average(client_weights)
        return self.global_weights

# if __name__ == "__main__":
#     from preprocessing.face_preprocessing import FacePreprocessor
#     from model.embedding_model import FaceEmbeddingModel
#     from federated.client import FederatedClient

#     preprocessor = FacePreprocessor()
#     embedder = FaceEmbeddingModel()

#     images, labels, names = preprocessor.load_lfw_data()

#     clients = [FederatedClient(i) for i in range(3)]
#     client_weights = []

#     for idx, client in enumerate(clients):
#         embeddings = []
#         local_labels = []

#         for i in range(idx * 50, (idx + 1) * 50):
#             face = preprocessor.detect_and_preprocess(images[i])
#             if face is not None:
#                 embeddings.append(embedder.get_embedding(face))
#                 local_labels.append(labels[i])

#         embeddings = np.array(embeddings)
#         local_labels = np.array(local_labels)

#         if embeddings.shape[0] == 0:
#             print(f"Client {idx} skipped due to no valid samples")
#             continue

#         if len(np.unique(local_labels)) < 2:
#             print(f"Client {idx} skipped due to single-class data")
#             continue

#         weights = client.train(embeddings, local_labels)
#         client_weights.append(weights)

#     server = FederatedServer()
#     global_weights = server.aggregate(client_weights)

#     print("Federated aggregation complete.")
#     print("Global weight shape:", global_weights.shape)