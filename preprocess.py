import os
import cv2
import argparse
from PIL import Image, ImageSequence
from tqdm import tqdm

def extract_frames_from_mp4(video_path, output_dir):
    """Extract frames from an MP4 video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (512, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
        count += 1
    cap.release()

def extract_frames_from_gif(gif_path, output_dir):
    """Extract frames from a GIF file."""
    with Image.open(gif_path) as im:
        count = 0
        for frame in ImageSequence.Iterator(im):
            frame = frame.convert("RGB").resize((512, 512))
            frame.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
            count += 1

def main(args):
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_files = [f for f in os.listdir(input_path)
                   if f.lower().endswith(('.mp4', '.gif'))]

    if not video_files:
        print("[WARNING] No MP4 or GIF files found in the input directory.")
        return

    for video_file in tqdm(video_files):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_path, video_file)
        save_dir = os.path.join(output_path, video_name)
        os.makedirs(save_dir, exist_ok=True)

        if video_file.lower().endswith('.mp4'):
            extract_frames_from_mp4(video_path, save_dir)
        elif video_file.lower().endswith('.gif'):
            extract_frames_from_gif(video_path, save_dir)
    print("All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video preprocessing script (supports MP4 and GIF).")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the folder containing videos.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save extracted frames.")
    args = parser.parse_args()
    main(args)

# python preprocess.py --input_path ./test/input --output_path ./test/output
