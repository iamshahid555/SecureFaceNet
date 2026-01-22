# SecureFaceNet

SecureFaceNet is a privacy-preserving facial recognition prototype that combines
Computer Vision, Federated Learning, and Differential Privacy to enable secure
biometric model training without centralized storage of raw facial data.

The system is designed to ensure that facial images never leave client devices.
Only privacy-protected model updates are shared with a central server, reducing
the risk of data breaches and improving compliance with data protection
regulations such as GDPR.

---

## Key Features

- Local face detection and preprocessing using OpenCV
- Face representation using pretrained FaceNet embeddings
- Local client-side model training
- Federated Learning using the FedAvg algorithm
- Differentially Private training using DP-SGD (Opacus)
- Formal privacy accounting with reported privacy budget (ε)
- Explicit handling of non-IID client data distributions

---

## Tech Stack

- Python 3.9+
- PyTorch
- OpenCV
- FaceNet (facenet-pytorch)
- Opacus (Differential Privacy)
- NumPy
- scikit-learn

---

## Project Structure

```
SecureFaceNet/
├── preprocessing/ # Face detection, cropping, normalization
├── model/ # Embedding extraction and classifiers
├── federated/ # Federated learning (client, server, FedAvg)
├── privacy/ # Differentially private training (DP-SGD)
├── requirements.txt # Project dependencies
└── README.md
```

---

## Installation & Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/iamshahid555/SecureFaceNet.git
cd SecureFaceNet

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Project (End-to-End Demo)

The main demonstration entry point is the differentially private training module.

Run the following command from the project root:

python -m privacy.dp_training

This command performs the complete pipeline: 1. Loads the LFW dataset locally 2. Detects and preprocesses faces using OpenCV 3. Extracts facial embeddings using FaceNet 4. Trains a classifier using DP-SGD 5. Computes and reports the privacy budget (ε)

Output:

```
Epoch 1, Loss: 1.1164
Epoch 2, Loss: 1.1143
Epoch 3, Loss: 1.1125
Epoch 4, Loss: 1.1115
Epoch 5, Loss: 1.1080
```

    Privacy budget ε = 11.49

## System Architecture

The overall architecture of SecureFaceNet is shown below. All sensitive operations
are performed locally on client devices, while only privacy-protected updates are
shared with the federated server.

<p align="center">
  <img src="diagrams/High-Level System Architecture of SecureFaceNet.png" width="650">
</p>

## Differential Privacy Workflow

<p align="center">
  <img src="diagrams/Differential Privacy Workflow.png" width="550">
</p>

## Privacy and Ethical Considerations

    •	Raw facial images are processed and retained only on the client side
    •	No biometric data is uploaded to the central server
    •	Federated Learning minimizes data exposure by sharing only model parameters
    •	Differential Privacy provides formal guarantees against individual data leakage
    •	The system design aligns with privacy-by-design and GDPR principles

## Academic Context

This project was developed as part of a Master’s-level Computer Science program.
It demonstrates applied knowledge of privacy-preserving machine learning,
distributed systems, and ethical AI design.

The implementation emphasizes clarity, correctness, and reproducibility over
production-scale optimization.

## Limitations and Future Work

    •	The current implementation simulates federated clients on a single machine
    •	Accuracy is reduced under differential privacy constraints
    •	Future work may include:
    •	Multi-round federated training
    •	Larger-scale client simulations
    •	Hyperparameter tuning for improved privacy–utility trade-offs
    •	Secure RNG mode for production-grade DP training

## License

This project is released for academic and educational use.
