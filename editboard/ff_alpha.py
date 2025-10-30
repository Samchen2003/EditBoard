import os 
import cv2
import numpy as np
from editboard.test_optflow import compute_optical_flow, apply_optical_flow
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

def get_warped_result_list(video_path, flow_list):
    warp_list = []
    frames = os.listdir(video_path)
    frames = [img for img in frames if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    frames.sort()
    for i in range(0,len(frames)-1):
        pp = os.path.join(video_path, frames[i])
        img1 = cv2.imread(pp)
        flow = flow_list[i]
        warped = apply_optical_flow(img1, flow)
        warp_list.append(warped)
    return warp_list

def calculate_ff_alpha(original,ori_warp,edit,edit_warp,threshold=5):
    m,n,_ = original.shape
    mask = np.zeros((m,n))

    diff = cv2.absdiff(original, ori_warp)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_edit = cv2.absdiff(edit, edit_warp)
    # diff_gray_edit = cv2.cvtColor(diff_edit, cv2.COLOR_BGR2GRAY)
    diff_gray_edit = np.max(diff_edit,-1)
    for i in range(m):
        for j in range(n):
            if diff_gray[i][j] <= threshold:
                mask[i][j] = 1
            else:
                mask[i][j] = 0

    percentage_of_valid_pixel = np.sum(mask==1)/512/512

    a = np.sum(np.multiply(mask,diff_gray_edit))
    result = a/np.sum(mask==1)
    return result, percentage_of_valid_pixel


def ff_alpha_for_video(original_video_path, edited_video_path, threshold = 5):
    result = []
    valid_percentage = []
    original_frames = os.listdir(original_video_path)
    original_frames = [img for img in original_frames if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    original_frames.sort()

    edited_frames = os.listdir(edited_video_path)
    edited_frames = [img for img in edited_frames if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    edited_frames.sort()

    flow_list = get_optical_flow_list(original_video_path)
    edit_warp_result = get_warped_result_list(edited_video_path,flow_list)
    original_warp_result = get_warped_result_list(original_video_path,flow_list)

    for i in range(0, len(edit_warp_result)):
        original = cv2.imread(os.path.join(original_video_path,original_frames[i+1]))
        ori_warp = original_warp_result[i]
        edit = cv2.imread(os.path.join(edited_video_path,edited_frames[i+1]))
        edit_warp = edit_warp_result[i]
        score, valid = calculate_ff_alpha(original, ori_warp, edit, edit_warp,threshold)
        result.append(score)
        valid_percentage.append(valid)

    if sum(valid_percentage)/len(valid_percentage) >= 0.70:
        return sum(result)/len(edit_warp_result)
    else:
        return 0

def compute_ff_alpha(json_dir, device, submodules_list):
    metadata = load_json(json_dir)
    result = {}
    for i in tqdm(metadata):
        score = ff_alpha_for_video(i["original_video_path"], i["edited_video_path"])
        result[i["original_video_path"] + i["edited_video_path"]] = score
    return result
    