import torch
import os
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from editboard.utils import load_video, load_dimension_info

def transform(images, preprocess_mode='shorter'):
    """preprocess_mode is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
    """
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def technical_quality(model, video_list, device):
    preprocess_mode = 'longer'
    video_results = {}
    for video_path in tqdm(video_list):
        images = load_video(video_path)
        images = transform(images, preprocess_mode)
        acc_score_video = 0.
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
        video_results[os.path.dirname(os.path.dirname(video_path))] = (acc_score_video/len(images)) / 100
    return video_results


def compute_imaging_quality(json_dir, device, submodules_list):
    model_path = submodules_list['model_path']

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    
    video_list = load_dimension_info(json_dir, dimension='imaging_quality')
    video_results = technical_quality(model, video_list, device)
    return video_results
