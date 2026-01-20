import numpy as np


def federated_average(weight_list):
    """
    Perform Federated Averaging (FedAvg).
    weight_list: list of numpy arrays (model weights from clients)
    """
    return np.mean(weight_list, axis=0)