import argparse
import json
import os
import numpy as np
from typing import Type
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector, FacenetDetector
from utils import get_video_paths, get_method_from_name
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def process_videos(videos, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = FacenetDetector(device)
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    for item in tqdm(loader, desc=f"Face Extract Progress:", colour="green"):
        result = {}
        video, indices, frames = item[0]
        out_dir = os.path.join(opt.data_path, "boxes", get_method_from_name(video))
        id = os.path.splitext(os.path.basename(video))[0]
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
        for j, frames in enumerate(batches):
            result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})
        os.makedirs(out_dir, exist_ok=True)
        if len(result) > 0:
            with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)
    if len(missed_videos) > 0:
        print("No faces detected in.")
        print(missed_videos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="FFPP", type=str,
                        help='Dataset (FFPP / CDF)')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument("--processes", help="Number of processes", default=10)
    opt = parser.parse_args()
    print(opt)

    dataset = 0 if opt.dataset.upper() == "CDF" else 1
    videos_paths = get_video_paths(opt.data_path, dataset)

    print("Video Numbers:", len(videos_paths), "Dataset:", opt.dataset)
    process_videos(videos_paths, opt)


if __name__ == "__main__":
    main()
