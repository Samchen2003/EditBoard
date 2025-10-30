# [AAAI 2025] EditBoard: Towards a Comprehensive Evaluation Benchmark for Text-Based Video Editing Models
[AAAI 2025] This is the official repo of the paper "EditBoard, a comprehensive evaluation benchmark for text-based video editing models" [[Paper]](https://arxiv.org/pdf/2409.09668).

### :book: Table of Contents
- [Installation](#installation)
- [Dataset structure](#data)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

<a name="installation"></a>
## :hammer: Installation

~~~bash
conda create -n EditBoard python==3.9
conda activate EditBoard
pip install -r requirements.txt
~~~


<a name="data"></a>
## :file_folder: Dataset Structure

For any given video, you need to segment it into frames and save all the frames into a directory named after the video. All frames must be resized to 512x512 pixels. To simplify this process, we provide a preprocessing script, `preprocess.py`, which supports MP4 and GIF video formats.

The command to run the script is:
```bash
python preprocess.py --input_path <path_to_your_videos> --output_path <path_to_save_frames>
```
- `--input_path`: The path to the directory containing your videos.
- `--output_path`: The path where the resulting frame directories will be saved.

Each frame folder will contain all frames from the corresponding video, e.g.:

```
dataset/
├── bear/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...
├── bear_white/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   └── ...
└── bear_mask/
    ├── frame_00000.png
    ├── frame_00001.png
    └── ...
```

:warning: **Important:**  
It is crucial that the corresponding original video, edited video, and semantic_mask folders contain the same number of image frames.


<a name="usage"></a>
## :rocket: Usage

We have implemented all nine evaluation dimensions used in our paper:
`["ff_alpha", "ff_beta", "semantic_score", "success_rate", "clip_similarity", 'subject_consistency', 'background_consistency', 'aesthetic_quality', 'imaging_quality']`

We offer two forms of commands for evaluation:
- **Normal Command** – evaluate one pair of videos at a time.  
- **Script Command** – evaluate multiple pairs in batch mode using a CSV or Excel file.

The final evaluation results will be saved in `{output_path}/{result_name}_eval_results.json`.


### Normal Command

This is a full example for evaluating all nine dimensions on a single pair of videos.

```bash
python -W ignore evaluate.py \
  --output_path './output/' \
  --result_name "result" \
  --dimension "ff_alpha" "ff_beta" "semantic_score" "success_rate" "clip_similarity" 'subject_consistency' 'background_consistency' 'aesthetic_quality' 'imaging_quality' \
  --original_video_path './sample/bear' \
  --edited_video_path './sample/bear_white' \
  --semantic_mask_path './sample/bear_mask' \
  --source_prompt 'a brown bear walks on rocks' \
  --target_prompt 'a white bear walks on rocks'
```


### Script Command

This command evaluates multiple pairs in batch mode using a CSV or Excel file. The `--dimension` and `--script` arguments are mandatory.

```bash
python -W ignore evaluate.py \
  --output_path './output/' \
  --result_name "result" \
  --dimension "ff_alpha" "ff_beta" "semantic_score" "success_rate" "clip_similarity" 'subject_consistency' 'background_consistency' 'aesthetic_quality' 'imaging_quality' \
  --script './sample/script.csv'
```

The script file (e.g., `--script`) must be a `.csv` or `.xlsx` file with the following header and format:
| original_video_path | edited_video_path | semantic_mask_path | source_prompt | target_prompt |
|----------------------|------------------|--------------------|----------------|----------------|
| ./sample/bear | ./sample/bear_autumn | ./sample/bear_mask | a brown bear walks on rocks | a brown bear walks on rocks in the autumn |

An example script file is available at `/EditBoard/sample/script.csv`.


### Required Inputs for Each Dimension

Different dimensions require different input fields.   Please ensure all necessary arguments are provided when running evaluation.

| Dimension | Required Inputs |
|------------|----------------|
| `ff_alpha`, `ff_beta` | `original_video_path`, `edited_video_path` |
| `semantic_score` | `original_video_path`, `edited_video_path`, `semantic_mask_path` |
| `success_rate`, `clip_similarity` | `edited_video_path`, `source_prompt`, `target_prompt` |
| `subject_consistency`, `background_consistency`, `aesthetic_quality`, `imaging_quality` | `edited_video_path` |

**Example Commands for Each Dimension**

- **`ff_alpha`, `ff_beta`**
  ```bash
  python -W ignore evaluate.py \
    --dimension "ff_alpha" "ff_beta" \
    --original_video_path './sample/bear' \
    --edited_video_path './sample/bear_white'
  ```

- **`semantic_score`**
  ```bash
  python -W ignore evaluate.py \
    --dimension "semantic_score" \
    --original_video_path './sample/bear' \
    --edited_video_path './sample/bear_white' \
    --semantic_mask_path './sample/bear_mask'
  ```

- **`success_rate`, `clip_similarity`**
  ```bash
  python -W ignore evaluate.py \
    --dimension "success_rate" "clip_similarity" \
    --edited_video_path './sample/bear_white' \
    --source_prompt 'a brown bear walks on rocks' \
    --target_prompt 'a white bear walks on rocks'
  ```

- **`subject_consistency`, `background_consistency`,  `aesthetic_quality`, `imaging_quality`**
  ```bash
  python -W ignore evaluate.py \
    --dimension "subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" \
    --edited_video_path './sample/bear_white'
  ```


<a name="acknowledgement"></a>
## :hearts: Acknowledgement

This project wouldn't be possible without the following open-sourced repositories: [CLIP](https://github.com/openai/CLIP) and [VBench](https://github.com/Vchitect/VBench).

<a name="citation"></a>
## :mailbox: Citation

If you find this repo useful for your research, please consider citing our work:

~~~
@inproceedings{chen2025editboard,
  title={Editboard: Towards a comprehensive evaluation benchmark for text-based video editing models},
  author={Chen, Yupeng and Chen, Penglin and Zhang, Xiaoyu and Huang, Yixian and Xie, Qian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={15},
  pages={15975--15983},
  year={2025}
}
~~~
