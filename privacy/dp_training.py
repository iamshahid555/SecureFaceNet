import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DPFaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_with_dp(embeddings, labels, epochs=5, lr=0.01):
    """
    Train a simple classifier with Differential Privacy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    unique_labels = torch.unique(y)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    y = torch.tensor([label_mapping[label.item()] for label in y], dtype=torch.long).to(device)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    model = DPFaceClassifier(
        input_dim=X.shape[1],
        num_classes=len(torch.unique(y))
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Privacy budget ε = {epsilon:.2f}")

    return model

if __name__ == "__main__":
    from preprocessing.face_preprocessing import FacePreprocessor
    from model.embedding_model import FaceEmbeddingModel

    preprocessor = FacePreprocessor()
    embedder = FaceEmbeddingModel()

    images, labels, _ = preprocessor.load_lfw_data()

    embeddings = []
    dp_labels = []

    for i in range(100):
        face = preprocessor.detect_and_preprocess(images[i])
        if face is not None:
            embeddings.append(embedder.get_embedding(face))
            dp_labels.append(labels[i])

    embeddings = np.array(embeddings)
    dp_labels = np.array(dp_labels)

    print("Training with Differential Privacy...")
    train_with_dp(embeddings, dp_labels)