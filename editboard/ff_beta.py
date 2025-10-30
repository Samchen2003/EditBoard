import os 
import cv2
import numpy as np
from editboard.test_optflow import compute_optical_flow
from editboard.utils import load_json
from tqdm import tqdm

def get_optical_flow_list(video_path):
    flow_list = []
    frames = os.listdir(video_path)
    frames = [img for img in frames if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    frames.sort()
    for i in range(0,len(frames)-1):
        img1 = cv2.imread(os.path.join(video_path, frames[i]))
        img2 = cv2.imread(os.path.join(video_path, frames[i+1]))
        flow = compute_optical_flow(img1,img2)
        flow_list.append(flow)
    return flow_list

##check
def ff_beta_for_one(a, b):
    return np.sum((1 - np.sum(a*b, -1) / ((np.sum(a*a, -1))**0.5 + 1e-7) /  ((np.sum(b*b, -1))**0.5 + 1e-7)) ) /(a.shape[0]*a.shape[1])
    # return np.sum((1 - np.sum(a*b, -1) / ((np.sum(a*a, -1))**0.5 + 1e-7) /  ((np.sum(b*b, -1))**0.5 + 1e-7)) * np.sum((a-b)**2,-1) ** 0.5) /(a.shape[0]*a.shape[1])

def ff_beta_for_video(original_video_path, edited_video_path):
    result = []

    flow_list_ori = get_optical_flow_list(original_video_path)
    flow_list_edit = get_optical_flow_list(edited_video_path)

    for i  in range(len(flow_list_edit)):
        flow1 = flow_list_ori[i]
        flow2 = flow_list_edit[i]
        result.append(ff_beta_for_one(flow1,flow2))
    return sum(result)/len(flow_list_edit)

def compute_ff_beta(json_dir, device, submodules_list):
    metadata = load_json(json_dir)
    result = {}
    for i in tqdm(metadata):
        score = ff_beta_for_video(i["original_video_path"], i["edited_video_path"])
        result[i["original_video_path"] + i["edited_video_path"]] = score
    return result