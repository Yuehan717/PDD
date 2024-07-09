# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from traceback import print_tb

from cv2 import transform

from basicsr.utils.logger import get_root_logger
from .augmentation import (BinarizeImage, ColorJitter, CopyValues, Flip,
                           GenerateFrameIndices,
                           GenerateFrameIndiceswithPadding,
                           GenerateSegmentIndices, MirrorSequence, Pad,
                           Quantize, RandomAffine, RandomJitter,
                           RandomMaskDilation, RandomTransposeHW, Resize,
                           TemporalReverse, UnsharpMasking)
from .compose import Compose
from .crop import (Crop, CropAroundCenter, CropAroundFg, CropAroundUnknown,
                   CropLike, FixedCrop, ModCrop, PairedRandomCrop,
                   RandomResizedCrop)
from .formating import (Collect, FormatTrimap, GetMaskedImage, ImageToTensor,
                        ToTensor)
from .generate_assistant import GenerateCoordinateAndCell, GenerateHeatmap
from .loading import (GetSpatialDiscountMask, LoadImageFromFile,
                      LoadImageFromFileList, LoadMask, LoadPairedImageFromFile,
                      RandomLoadResizeBg)
from .matlab_like_resize import MATLABLikeResize
from .matting_aug import (CompositeFg, GenerateSeg, GenerateSoftSeg,
                          GenerateTrimap, GenerateTrimapWithDistTransform,
                          MergeFgAndBg, PerturbBg, TransformTrimap)
from .normalization import Normalize, RescaleToZeroOne
from .random_degradations import (DegradationsWithShuffle, RandomBlur,
                                  RandomJPEGCompression, RandomNoise,
                                  RandomResize, RandomVideoCompression, DegradationsWithSampling, DoubleDegradationsWithSampling)
from.random_degradations_labeled import (RandomBlurLabeled, RandomNoiseLabeled, 
                                         RandomResizeLabeled, RandomJPEGCompressionLabeled, DegradationsWithSamplingLabeled)
from.random_degradations_BinaryLabeled import (RandomBlurBinaryLabeled, RandomNoiseBinaryLabeled, 
                                         RandomResizeBinaryLabeled, RandomJPEGCompressionBinaryLabeled, DegradationsWithSamplingBinaryLabeled)
from .random_down_sampling import RandomDownSampling
from basicsr.utils.registry import PIPELINE_REGISTRY

__all__ = [
    'Collect', 'FormatTrimap', 'LoadImageFromFile', 'LoadMask',
    'RandomLoadResizeBg', 'Compose', 'ImageToTensor', 'ToTensor',
    'GetMaskedImage', 'BinarizeImage', 'Flip', 'Pad', 'RandomAffine',
    'RandomJitter', 'ColorJitter', 'RandomMaskDilation', 'RandomTransposeHW',
    'Resize', 'RandomResizedCrop', 'CenterCrop', 'Crop', 'CropAroundCenter',
    'CropAroundUnknown', 'ModCrop', 'PairedRandomCrop', 'Normalize',
    'RescaleToZeroOne', 'GenerateTrimap', 'MergeFgAndBg', 'CompositeFg',
    'TemporalReverse', 'LoadImageFromFileList', 'GenerateFrameIndices',
    'GenerateFrameIndiceswithPadding', 'FixedCrop', 'LoadPairedImageFromFile',
    'GenerateSoftSeg', 'GenerateSeg', 'PerturbBg', 'CropAroundFg',
    'GetSpatialDiscountMask', 'RandomDownSampling',
    'GenerateTrimapWithDistTransform', 'TransformTrimap',
    'GenerateCoordinateAndCell', 'GenerateSegmentIndices', 'MirrorSequence',
    'CropLike', 'GenerateHeatmap', 'MATLABLikeResize', 'CopyValues',
    'Quantize', 'RandomBlur', 'RandomJPEGCompression', 'RandomNoise',
    'DegradationsWithShuffle', 'RandomResize', 'UnsharpMasking',
    'RandomVideoCompression', 'CropSequence', 'DegradationsWithSampling',
    'RandomBlurLabeled', 'RandomNoiseLabeled', 'RandomResizeLabeled',
    'RandomJPEGCompressionLabeled', 'DegradationsWithSamplingLabeled',
    'RandomBlurBinaryLabeled', 'RandomNoiseBinaryLabeled', 'RandomResizeBinaryLabeled',
    'RandomJPEGCompressionBinaryLabeled', 'DegradationsWithSamplingBinaryLabeled', 'DoubleDegradationsWithSampling'
]

class BuildPipeline:
    """
    Build degradation pipline.
    """
    def __init__(self,pipeline_opts):
        pipeline_opts = deepcopy(pipeline_opts)
        self.pipelines = []
        for pipeline_opt in pipeline_opts:
            if isinstance(pipeline_opt, dict):
                # print(f"pipline {pipeline_opt}")
                t_type = pipeline_opt.pop('type')
                transform = PIPELINE_REGISTRY.get(t_type)(**pipeline_opt)
                self.pipelines.append(transform)
            else:
                raise TypeError(f'transform must be a dict, '
                                f'but got {type(pipeline_opt)}')
        logger = get_root_logger()
        logger.info(f'Pipeline is built')

    def __call__(self, data):
        # print(stage)
        for t in self.pipelines:
            data = t(data)
            if data is None:
                return None
        return data

class BuildPipelineWithLabels:
    """
    Build degradation pipline.
    """
    def __init__(self,pipeline_opts):
        pipeline_opts = deepcopy(pipeline_opts)
        self.pipelines = []
        for pipeline_opt in pipeline_opts:
            if isinstance(pipeline_opt, dict):
                # print(f"pipline {pipeline_opt}")
                t_type = pipeline_opt.pop('type')
                transform = PIPELINE_REGISTRY.get(t_type)(**pipeline_opt)
                self.pipelines.append(transform)
            else:
                raise TypeError(f'transform must be a dict, '
                                f'but got {type(pipeline_opt)}')
        logger = get_root_logger()
        logger.info(f'Pipeline is built')

    def __call__(self, data):
        # print(stage)
        labels_list = []
        for t in self.pipelines:
            data = t(data)
            if data is None:
                return None
            if isinstance(data, tuple):
                data, labels = data
                if isinstance(labels[0], list):
                    labels_list.extend(labels)
                else:
                    labels_list.append(labels)
        return data, labels_list


