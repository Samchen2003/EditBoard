import cv2
import os
import numpy as np
from editboard.utils import load_json
from tqdm import tqdm

def readimagefile(filepath):
    frames = os.listdir(filepath)
    frames = [img for img in frames if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    frames.sort()
    return frames

def semantic_score(original_file, edit_file, mask_file, res=512):
    result = []
    mask_frame = readimagefile(mask_file)
    original_frame = readimagefile(original_file)
    edit_frame = readimagefile(edit_file)
    for i in range(len(mask_frame)):
        mask = cv2.imread(os.path.join(mask_file, mask_frame[i]))
        
        original = cv2.imread(os.path.join(original_file, original_frame[i]))
        edit = cv2.imread(os.path.join(edit_file, edit_frame[i]))

        diff = cv2.absdiff(original, edit)
        diff = np.max(diff, -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask_0_1 = np.zeros((res,res))
        for i in range(res):
            for j in range(res):
                if mask[i][j] == 0:
                    mask_0_1[i][j] = 1
                else:
                    mask_0_1[i][j] = 0

        a = np.sum(np.multiply(mask_0_1,diff))
        result_frame = a/np.sum(mask_0_1==1)

        result.append(result_frame)
    return sum(result)/len(original_frame)

def compute_semantic_score(json_dir, device, submodules_list):
    metadata = load_json(json_dir)
    result = {}
    for i in tqdm(metadata):
        score = semantic_score(i["original_video_path"], i["edited_video_path"], i["semantic_mask_path"])
        result[i["original_video_path"] + i["edited_video_path"] + i["semantic_mask_path"]] = score
    return result
