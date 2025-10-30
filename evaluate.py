# export CUDA_VISIBLE_DEVICES=0,1
# python -W ignore evaluate.py  --dimension 'subject_consistency' 'background_consistency' 'aesthetic_quality' 'imaging_quality'  --edited_video_path './sample/test' 

# python -W ignore evaluate.py  --dimension 'subject_consistency' 'background_consistency' 'aesthetic_quality' 'imaging_quality' "ff_alpha" "ff_beta" "semantic_score" "clip_similarity" "success_rate" --original_video_path './sample/bear' --edited_video_path './sample/bear_white' --semantic_mask_path './sample/bear_mask' --source_prompt 'a brown bear walks on rocks' --target_prompt 'a white bear walks on rocks'
# python -W ignore evaluate.py  --dimension 'subject_consistency' 'background_consistency' 'aesthetic_quality' 'imaging_quality' "ff_alpha" "ff_beta" "semantic_score" "clip_similarity" "success_rate" --script './script.csv'
# ff_alpha !
# ff_beta !
# semantic_score !
# clip_similarity
# success_rate

import torch
import os
from editboard import EditBoard
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='EditBoard', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./output/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default = "result"
    )

    parser.add_argument(
        "--original_video_path",
        type=str,
        help="folder that contains all frames of the original video",
        default=None
    )
    parser.add_argument(
        "--edited_video_path",
        type=str,
        help="folder that contains all frames of the edited video",
        default=None
    )
    parser.add_argument(
        "--semantic_mask_path",
        type=str,
        help="folder that contains the semantic mask",
        default=None
    )
    parser.add_argument(
        "--source_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="csv or excel are both fine"
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    os.makedirs(args.output_path, exist_ok=True)
    my_EditBoard = EditBoard(device, args.output_path)
    
    print(f'Start EditBoard Evaluation!')

    my_EditBoard.evaluate(
        original_video_path = args.original_video_path,
        edited_video_path = args.edited_video_path,
        semantic_mask_path = args.semantic_mask_path,
        source_prompt = args.source_prompt,
        target_prompt = args.target_prompt,

        dimension_list = args.dimension,
        name = args.result_name,
        script = args.script
    )


if __name__ == "__main__":
    main()
