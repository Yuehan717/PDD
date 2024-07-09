import numpy as np
import torch.nn.functional as F
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_mse(predictions, labels,  **kwargs):

    mse = (predictions - labels)**2
    mse = np.mean(mse.numpy())
    return mse