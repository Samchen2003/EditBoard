import os

from .utils import init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path
import shutil
from PIL import Image
import pandas as pd

def frames2gif(source_folder):
    output_folder = os.path.join(source_folder, "tempt_dir")

    os.makedirs(output_folder, exist_ok=True)
    
    images = []
    
    for file_name in sorted(os.listdir(source_folder)):
        file_path = os.path.join(source_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(file_path)
            images.append(img)
            # print(file_name)
    
    if images:
        folder_name = os.path.basename(source_folder)
        gif_path = os.path.join(output_folder, f"{folder_name}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
        
        for img in images:
            img.close()
    else:
        raise Exception("No images found in the source folder.")
    
    return output_folder

class EditBoard(object):
    def __init__(self, device, output_path):
        self.device = device                        # cuda or cpu
        self.output_path = output_path              # output directory to save EditBoard results
        os.makedirs(self.output_path, exist_ok=True)

    def build_metadata_json_single(
            self, original_video_path, edited_video_path, semantic_mask_path,
            source_prompt, target_prompt,
            dimension_list, name
        ):
        cur_full_info_list=[]
        
        temp = {
            k: v for k, v in {
                "original_video_path": original_video_path,
                "edited_video_path": edited_video_path,
                "semantic_mask_path": semantic_mask_path,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "dimension": dimension_list,
            }.items() if v is not None
        }
        
        cur_full_info_list.append(temp)

        cur_full_info_path = os.path.join(self.output_path, name+'_metadata.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation metadata saved to {cur_full_info_path}')
        return cur_full_info_path

    def build_metadata_json_multi(self, dimension_list, name, script):
        cur_full_info_list = []

        if script.split(".")[-1] == 'xlsx':
            df = pd.read_excel(script)
        elif script.split(".")[-1] == 'csv':
            df = pd.read_csv(script)  
        else:
            raise Exception("Prompt file must be excel or csv!")

        available_columns = set(df.columns)
        
        expected_columns = {
            "original_video_path": "original_video_path",
            "edited_video_path": "edited_video_path", 
            "semantic_mask_path": "semantic_mask_path",
            "source_prompt": "source_prompt",
            "target_prompt": "target_prompt"
        }
        
        for index, row in df.iterrows():
            temp = {}
            
            for col_key, json_key in expected_columns.items():
                if col_key in available_columns and pd.notna(row[col_key]):
                    temp[json_key] = row[col_key]
            
            temp["dimension"] = dimension_list
                
            cur_full_info_list.append(temp)

        cur_full_info_path = os.path.join(self.output_path, name + '_metadata.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation metadata saved to {cur_full_info_path}')
        return cur_full_info_path

    def evaluate(
            self, original_video_path, edited_video_path, semantic_mask_path,
            source_prompt, target_prompt,
            dimension_list, name, script
        ):
        read_frame = False
        results_dict = {}
        if dimension_list is None:
            raise Exception("Dimension can't be none!")
        submodules_dict = init_submodules(dimension_list, read_frame=read_frame)

        if script == None:
            print("Using Normal Command!")
            cur_full_info_path = self.build_metadata_json_single(
                original_video_path, edited_video_path, semantic_mask_path,
                source_prompt, target_prompt,
                dimension_list, name
            )
        else:
            print("Using Script Command!")
            cur_full_info_path = self.build_metadata_json_multi(
                dimension_list, name, script
            )
        

        # Start calculating
        flag = False
        metadata = load_json(cur_full_info_path) 
        gif_list = [] 
        if any(dimension in dimension_list for dimension in ['subject_consistency', 'background_consistency', 'aesthetic_quality', 'imaging_quality']):
            flag = True
            for i in metadata:
                gif_path = frames2gif(i["edited_video_path"])
                gif_list.append(gif_path)

        for dimension in dimension_list:
            print(f"Calculating {dimension} ...")
            try:
                dimension_module = importlib.import_module(f'editboard.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            # print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list)
            results_dict[dimension] = results

        if flag:
            for i in gif_list:
                shutil.rmtree(i)
        # Finish calculating
        
        for i in metadata:
            i["dimension"] = dict()
            for dimension in dimension_list:
                if dimension in ['subject_consistency', 'background_consistency', 'aesthetic_quality', 'imaging_quality']:
                    i["dimension"][dimension] = results_dict[dimension][i["edited_video_path"]]
                elif dimension in ["ff_alpha", "ff_beta"]:
                    i["dimension"][dimension] = results_dict[dimension][i["original_video_path"] + i["edited_video_path"]]
                elif dimension in ["clip_similarity", "success_rate"]:
                    i["dimension"][dimension] = results_dict[dimension][i["edited_video_path"] + i["source_prompt"] + i["target_prompt"]]
                elif dimension in ["semantic_score"]:
                    i["dimension"][dimension] = results_dict[dimension][i["original_video_path"] + i["edited_video_path"] + i["semantic_mask_path"]]
                else:
                    raise Exception("Wrong dimension!")

        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(metadata, output_name)
        print('All Done!')
        print(f'Evaluation results saved to {output_name}')
