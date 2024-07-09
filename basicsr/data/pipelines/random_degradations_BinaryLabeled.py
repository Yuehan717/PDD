# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import random

import cv2
import numpy as np

from mmagic.datasets.transforms import blur_kernels as blur_kernels
from basicsr.utils.registry import PIPELINE_REGISTRY

try:
    import av
    has_av = True
except ImportError:
    has_av = False

Label_ids = {
    'bicubic': 0,
    'area': 1,
    'bilinear': 2,
    'blur_iso_s1': 3,
    'blur_iso_s2': 4,
    'blur_iso_s3': 5,
    'blur_sinc_s1': 6,
    'blur_sinc_s2': 7,
    'blur_sinc_s3': 8,
    'noise_gaussian_s1':9,
    'noise_gaussian_s2':10,
    'noise_gaussian_s3':11,
    'noise_poisson_s1':12,
    'noise_poisson_s2':13,
    'noise_poisson_s3':14,
    'jpeg_q1':15,
    'jpeg_q2':16,
    'jpeg_q3':17
}

@PIPELINE_REGISTRY.register()
class RandomBlurBinaryLabeled:
    """Apply random blur to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def get_kernel(self, num_kernels):
        labels = []
        kernel_type = np.random.choice(
            self.params['kernel_list'], p=self.params['kernel_prob'])
        kernel_size = random.choice(self.params['kernel_size'])

        sigma_range = self.params.get('sigma', [0, 0])
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        sigma_step = self.params.get('sigma_step', 0)

        rotate_angle_range = self.params.get('rotate_angle', [-np.pi, np.pi])
        rotate_angle = np.random.uniform(rotate_angle_range[0],
                                         rotate_angle_range[1])
        rotate_angle_step = self.params.get('rotate_angle_step', 0)

        beta_gau_range = self.params.get('beta_gaussian', [0.5, 4])
        beta_gau = np.random.uniform(beta_gau_range[0], beta_gau_range[1])
        beta_gau_step = self.params.get('beta_gaussian_step', 0)

        beta_pla_range = self.params.get('beta_plateau', [1, 2])
        beta_pla = np.random.uniform(beta_pla_range[0], beta_pla_range[1])
        beta_pla_step = self.params.get('beta_plateau_step', 0)

        omega_range = self.params.get('omega', None)
        omega_step = self.params.get('omega_step', 0)
        if omega_range is None:  # follow Real-ESRGAN settings if not specified
            if kernel_size < 13:
                omega_range = [np.pi / 3., np.pi]
            else:
                omega_range = [np.pi / 5., np.pi]
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # determine blurring kernel
        kernels = []
        for _ in range(0, num_kernels):
            kernel = blur_kernels.random_mixed_kernels(
                [kernel_type],
                [1],
                kernel_size,
                [sigma, sigma],
                [sigma, sigma],
                [rotate_angle, rotate_angle],
                [beta_gau, beta_gau],
                [beta_pla, beta_pla],
                [omega, omega],
                None,
            )
            kernels.append(kernel)
            s = int((sigma + 1)//1.) if sigma!=3 else 3
            labels.append((Label_ids[f'blur_{kernel_type}_s{s}'], 1))

            # update kernel parameters
            sigma += np.random.uniform(-sigma_step, sigma_step)
            rotate_angle += np.random.uniform(-rotate_angle_step,
                                              rotate_angle_step)
            beta_gau += np.random.uniform(-beta_gau_step, beta_gau_step)
            beta_pla += np.random.uniform(-beta_pla_step, beta_pla_step)
            omega += np.random.uniform(-omega_step, omega_step)

            sigma = np.clip(sigma, sigma_range[0], sigma_range[1])
            rotate_angle = np.clip(rotate_angle, rotate_angle_range[0],
                                   rotate_angle_range[1])
            beta_gau = np.clip(beta_gau, beta_gau_range[0], beta_gau_range[1])
            beta_pla = np.clip(beta_pla, beta_pla_range[0], beta_pla_range[1])
            omega = np.clip(omega, omega_range[0], omega_range[1])

        return kernels, labels

    def _apply_random_blur(self, imgs, seq_prob):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # get kernel and blur the input
        kernels, labels = self.get_kernel(num_kernels=len(imgs))
        # imgs = [
        #     cv2.filter2D(img, -1, kernel)
        #     for img, kernel in zip(imgs, kernels)
        # ]
        outputs = []
        for img, kernel in zip(imgs, kernels):
            if np.random.uniform() > seq_prob:
                outputs.append(img)
            else:
                outputs.append(cv2.filter2D(img, -1, kernel))

        if is_single_image:
            outputs = outputs[0]
            labels = labels[0]

        return outputs, labels


    def __call__(self, results):
        for key in self.keys:
            results[key], labels = self._apply_random_blur(results[key], self.params.get('seq_prob', 1))

        return results, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        # repr_str += ', skiphalf'
        return repr_str


@PIPELINE_REGISTRY.register()
class RandomNoiseBinaryLabeled:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_gaussian_noise(self, imgs, seq_prob):
        labels=[]
        sigma_range = self.params['gaussian_sigma']
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.

        sigma_step = self.params.get('gaussian_sigma_step', 0)

        # gray_noise_prob = self.params['gaussian_gray_noise_prob']
        # is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            if np.random.uniform() > seq_prob:
                outputs.append(img)
            else:
                noise = np.float32(np.random.randn(*(img.shape))) * sigma
                # if is_gray_noise:
                #     noise = noise[:, :, :1]
                outputs.append(np.clip(img + noise, 0, 1))
                s = int(sigma//10.+1.) if sigma!=30 else 3
                labels.append((Label_ids[f'noise_gaussian_s{s}'], 1))
                # update noise level
                sigma += np.random.uniform(-sigma_step, sigma_step) / 255.
                sigma = np.clip(sigma, sigma_range[0] / 255.,
                                sigma_range[1] / 255.)

        return outputs, labels

    def _apply_poisson_noise(self, imgs, seq_prob):
        labels = []
        scale_range = self.params['poisson_scale']
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get('poisson_scale_step', 0)

        # gray_noise_prob = self.params['poisson_gray_noise_prob']
        # is_gray_noise = np.random.uniform() < gray_noise_prob

        outputs = []
        for img in imgs:
            if np.random.uniform() > seq_prob:
                outputs.append(img)
            else:
                noise = img.copy()
                # if is_gray_noise:
                #     noise = cv2.cvtColor(noise[..., [2, 1, 0]].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                #     noise = noise[..., np.newaxis]
                noise = np.clip((noise * 255.0).round(), 0, 255) / 255.
                unique_val = 2**np.ceil(np.log2(len(np.unique(noise))))
                noise = np.random.poisson(noise * unique_val) / unique_val - noise

                outputs.append(np.clip(img + noise * scale,0,1))
                s = int(scale//1.+1) if scale!=3 else 3
                labels.append((Label_ids[f'noise_poisson_s{s}'], 1))

                # update noise level
                scale += np.random.uniform(-scale_step, scale_step)
                scale = np.clip(scale, scale_range[0], scale_range[1])

        return outputs, labels

    def _apply_random_noise(self, imgs, seq_prob):
        noise_type = np.random.choice(
            self.params['noise_type'], p=self.params['noise_prob'])

        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        if noise_type.lower() == 'gaussian':
            imgs, labels = self._apply_gaussian_noise(imgs,seq_prob)
        elif noise_type.lower() == 'poisson':
            imgs, labels = self._apply_poisson_noise(imgs, seq_prob)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is '
                                      'not implemented.')

        if is_single_image:
            imgs = imgs[0]
            labels = labels[0]

        return imgs, labels
    
    def __call__(self, results):
        for key in self.keys:
            results[key], labels = self._apply_random_noise(results[key], self.params.get('seq_prob', 1))

        return results, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        # repr_str += ', skiphalf'
        
        return repr_str


@PIPELINE_REGISTRY.register()
class RandomJPEGCompressionBinaryLabeled:
    """Apply random JPEG compression to the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

    def _apply_random_compression(self, imgs, seq_prob):
        labels= []
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        # determine initial compression level and the step size
        quality = self.params['quality']
        quality_step = self.params.get('quality_step', 0)
        jpeg_param = round(np.random.uniform(quality[0], quality[1]))

        # apply jpeg compression
        outputs = []
        for img in imgs:
            if np.random.uniform() > seq_prob:
                outputs.append(img)
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
                _, img_encoded = cv2.imencode('.jpg', img * 255., encode_param)
                outputs.append(np.float32(cv2.imdecode(img_encoded, 1)) / 255.)
                q = int((jpeg_param-30)//20+1) if jpeg_param!=90 else 3
                labels.append((Label_ids[f'jpeg_q{q}'], 1))
                # update compression level
                jpeg_param += np.random.uniform(-quality_step, quality_step)
                jpeg_param = round(np.clip(jpeg_param, quality[0], quality[1]))

        if is_single_image:
            outputs = outputs[0]
            labels = labels[0]

        return outputs, labels

    def __call__(self, results):

        for key in self.keys:
            results[key], labels = self._apply_random_compression(results[key], self.params.get('seq_prob', 1))

        return results, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        # repr_str += ', skiphalf'
        return repr_str



@PIPELINE_REGISTRY.register()
class RandomResizeBinaryLabeled:
    """Randomly resize the input.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params

        self.resize_dict = dict(
            bilinear=cv2.INTER_LINEAR,
            bicubic=cv2.INTER_CUBIC,
            area=cv2.INTER_AREA)

    def _random_resize(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        h, w = imgs[0].shape[:2]

        resize_opt = self.params['resize_opt']
        resize_prob = self.params['resize_prob']
        nresize_prob = []
        for i in resize_prob:
            if isinstance(i,str):
                i = int(i.split('/')[0]) / int(i.split('/')[1])
            nresize_prob.append(i)
        resize_opt = np.random.choice(resize_opt, p=nresize_prob).lower()
        if resize_opt not in self.resize_dict:
            raise NotImplementedError(f'resize_opt [{resize_opt}] is not '
                                      'implemented')
        resize_opt_str = resize_opt
        resize_opt = self.resize_dict[resize_opt]

        # determine the target size, if not provided
        target_size = self.params.get('target_size', None)

        # resize the input
        outputs = [
            cv2.resize(img, target_size[::-1], interpolation=resize_opt)
            for img in imgs
        ]
        labels = [
            (Label_ids[resize_opt_str],1.) for _ in range(len(imgs))
        ]

        if is_single_image:
            outputs = outputs[0]
            labels = labels[0]

        return outputs, labels

    def __call__(self, results):

        for key in self.keys:
            results[key], labels = self._random_resize(results[key])

        return results, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str

allowed_degradations = {
    'RandomBlurBinaryLabeled': RandomBlurBinaryLabeled,
    'RandomNoiseBinaryLabeled': RandomNoiseBinaryLabeled,
    'RandomJPEGCompressionBinaryLabeled': RandomJPEGCompressionBinaryLabeled,
}


@PIPELINE_REGISTRY.register()
class DegradationsWithSamplingBinaryLabeled:
    """Apply random degradations to input, with degradations being sampled.

    Modified keys are the attributed specified in "keys".

    Args:
        degradations (list[dict]): The list of degradations.
        keys (list[str]): A list specifying the keys whose values are
            modified.
        N_samplings: number of samples, execute in random order
    Return:
        results: list of processed images
        labels: [[(a,b)...] , [(c,d).....]] list of label list 
    """

    def __init__(self, degradations, keys, n_samplings=None):

        self.keys = keys

        self.degradations = self._build_degradations(degradations)
        self.n_samplings = n_samplings
        self.N = len(degradations)

    def _build_degradations(self, degradations):
        for i, degradation in enumerate(degradations):
            degradation_ = allowed_degradations[degradation['type']]
            degradations[i] = degradation_(degradation['params'],
                                               self.keys)

        return degradations

    def __call__(self, results):
        labels_list = []
        # shuffle degradations
        n_sample = np.random.randint(self.n_samplings+1)
        if n_sample == 0:
            return results
        idx_list = list(range(0, self.N))
        np.random.shuffle(idx_list)
        idx_list = idx_list[:self.n_samplings]

        # apply degradations to input
        for i,degradation in enumerate(self.degradations):
            if i in idx_list:
                results, labels = degradation(results)
                labels_list.append(labels)

        return results, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(degradations={self.degradations}, '
                     f'keys={self.keys}, '
                     f'n_sampling={self.n_samplings})')
        repr_str += ', skiphalf'
        return repr_str