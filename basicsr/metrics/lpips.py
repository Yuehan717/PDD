import cv2
import numpy as np
import torch
import basicsr.metrics.LPIPS as LPIPS
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, **kwargs):
    """Calculate LPIPS (Full-reference perceptual metrics).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border: crop the measured region.

    Returns:
        float: lpips result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    model = LPIPS.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
    img = img / 127.5 - 1
    img2 = img2 / 127.5 - 1
    
    dist = model.forward(torch.from_numpy(img).unsqueeze(0).contiguous().permute(0,3,1,2).cuda().float(), 
                         torch.from_numpy(img2).unsqueeze(0).contiguous().permute(0,3,1,2).cuda().float())
    return dist.detach().squeeze().cpu()
    # pass
    # return lpips