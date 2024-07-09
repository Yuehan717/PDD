# from torch.utils.data import default_collate
import torch

def collate_mulsrc_fn(data):
    """
    data is a list of dict()
    """
    result = dict()
    for d in data:
        for key in list(d.keys()):
            if not (key in result):
                result[key] = [d[key]]
            else:
                result[key].append(d[key])
    for key in list(result.keys()):
        if isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key], dim=0)
    return result
