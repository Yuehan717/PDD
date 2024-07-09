# import flow_vis
import argparse
import cv2
import glob
import os
import shutil
import torch
import yaml
from collections import OrderedDict
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img
from basicsr.archs.bsrgan_arch import BSRGANRRDBNet
from basicsr.archs.rrdbnet_arch import RRDBNet

def func_color_correction(lr, hr):
    ### B, c, h, w
    _,c, h, w = hr.shape
    lr = lr.flatten(-2,-1)
    hr = hr.flatten(-2,-1)
    # print(hr.shape)
    miu_lr = torch.mean(lr, dim=-1, keepdim=True) # 1, 3, 1
    std_lr = torch.std(lr, dim=-1, keepdim=True) # 1, 3, 1
    miu_hr = torch.mean(hr, dim=-1, keepdim=True) # 1, 3, 1
    std_hr = torch.std(hr, dim=-1, keepdim=True) # 1, 3, 1
    
    hr_corrected = ((hr - miu_hr)/std_hr)*std_lr + miu_lr
    
    return hr_corrected.reshape(-1, c, h, w)



def inference(imgs, imgnames, model, save_path, color_correction=False):
    with torch.no_grad():
        outputs = model(imgs)
    
    if color_correction:
        outputs = func_color_correction(imgs, outputs)
    outputs = list(outputs)
    
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), output)

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def inference_vid(args, input_path, save_path, device, model, use_ffmpeg):
    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.to(device)
        inference(imgs, imgnames, model, save_path, args.color_corr)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.to(device)
            inference(imgs, imgnames, model, save_path, args.color_corr)
            print(f"{idx}/{num_imgs}")

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model', type=str, default='esrgan')
    parser.add_argument('--input_path', type=str, default='datasets/RealSR/Canon/test/4', help='input test image folder')
    parser.add_argument('--gt_path', type=str, default='datasets/RealSR/Canon/test/4', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/Canon', help='save image path')
    parser.add_argument('--interval', type=int, default=1, help='interval size')
    parser.add_argument('--mode', type=str, default='whole')
    parser.add_argument('--color_corr', action='store_true')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = args.model.lower()
    if model_name == 'esrgan':
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
    
    elif model_name == 'bsrgan':
        model = BSRGANRRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    else:
        print("Model not defined.")
    state = torch.load(args.model_path)
    if 'params' in list(state.keys()):
        state = torch.load(args.model_path)['params']
    elif 'params_ema' in list(state.keys()):
        state = torch.load(args.model_path)['params_ema']
    else:
        state = torch.load(args.model_path)
    
    model.load_state_dict(state, strict=True)

    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    input_path = args.input_path
    
    use_ffmpeg = False
    ### input_path is a dictionary
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    if not os.path.isdir(imgs_list[0]):
        num_imgs = len(imgs_list)
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.to(device)
            inference(imgs, imgnames, model, args.save_path, args.color_corr)#noise_list=noise_list[idx].to(device)
    else:
        video_names = os.listdir(input_path)
        video_names.sort()
        for video_name in video_names:
            video_path = os.path.join(input_path, video_name)
            save_path = os.path.join(args.save_path, video_name)
            os.makedirs(save_path, exist_ok=True)
            imgs_list = sorted(glob.glob(os.path.join(video_path, '*')))
            
            if os.path.isdir(imgs_list[0]):
                real_video_names = os.listdir(video_path)
                for real_video_name in real_video_names:
                    real_video_path = os.path.join(video_path, real_video_name)
                    real_save_path = os.path.join(args.save_path, video_name, real_video_name)
                    os.makedirs(real_save_path, exist_ok=True)
                    inference_vid(args, real_video_path, real_save_path, device, model, use_ffmpeg)
            # print(save_path)
            else:
                inference_vid(args, video_path, save_path, device, model, use_ffmpeg)

if __name__ == '__main__':
    main()