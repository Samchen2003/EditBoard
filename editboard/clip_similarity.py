import torch
import clip
from PIL import Image
from glob import glob
import numpy as np
import os
from editboard.utils import load_json
from tqdm import tqdm

def crop_read_image_path(image_path):
    origin_image = Image.open(image_path)
    w, h = origin_image.size
    if h > w:
        origin_image = origin_image.crop((0, h-w, w, h))
    return origin_image

def edit_success(image_path, source_prompt,target_prompt, model, preprocess, device):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)

    text = clip.tokenize([source_prompt, target_prompt]).to(device)
    target = clip.tokenize(target_prompt).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        target_features = model.encode_text(target)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()


        image_features = image_features.cpu().numpy()
        target_features = target_features.cpu().numpy()
        image_features_normalized = image_features / np.linalg.norm(image_features)
        text_features_normalized = target_features / np.linalg.norm(target_features)
    
    # Compute the cosine similarity
    image_features_normalized = image_features_normalized
    text_features_normalized = text_features_normalized

    similarity = np.sum(image_features_normalized * text_features_normalized, -1)

    if probs[0,1] >= probs[0,0]:
        return 1, similarity[0]
    
    else:
        return 0, similarity[0]

def video_score(edited_video_path, source_prompt, target_prompt, model, preprocess, device):
    count = 0
    score = 0
    file_list = os.listdir(edited_video_path)
    file_list = [img for img in file_list if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]

    for i in file_list:
        image_path = os.path.join(edited_video_path, i)
        count_sub, score_sub = edit_success(image_path, source_prompt, target_prompt, model, preprocess, device)
        count+=count_sub
        score+=score_sub

    success_rate = count/len(file_list)
    clip_similarity = score/len(file_list)
    
    return clip_similarity

def compute_clip_similarity(json_dir, device, submodules_list):
    model, preprocess = clip.load("ViT-B/32", device=device)

    metadata = load_json(json_dir)
    result = {}
    for i in tqdm(metadata):
        score = video_score(i["edited_video_path"], i["source_prompt"], i["target_prompt"], model, preprocess, device)
        result[i["edited_video_path"] + i["source_prompt"] + i["target_prompt"]] = score
    return result