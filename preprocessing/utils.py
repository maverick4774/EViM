import json
import os
from glob import glob
from pathlib import Path
import cv2
banned_folders = ["boxes", "set", "splits", "actors", "crops", "DeepFakeDetection", "actors", "zip", "find_missing.py", "txt", "copytest.py", "List_of_testing_videos.txt", "Celeb-DF-v2.zip"]
def get_video_paths(data_path, dataset, excluded_videos=[]):
    videos_folders = os.listdir(data_path)
    print("Video Path：")
    video_paths = []
    for folder in os.listdir(data_path):
        if any(banned in folder for banned in banned_folders):
            continue

        folder_path = os.path.join(data_path, folder)

        if dataset == 0:  # Celeb-DF
            video_paths.extend(
                os.path.join(folder_path, video)
                for video in os.listdir(folder_path)
            )
            print(folder_path)

        elif dataset == 1:  # FaceForensics++
            for sub_dir in os.listdir(folder_path):
                video_dir = os.path.join(folder_path, sub_dir, "c23", "videos")
                video_paths.extend(
                    os.path.join(video_dir, video)
                    for video in os.listdir(video_dir)
                )
                print(video_dir)

    return video_paths

def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)

    return originals_v if basename else originals


        
def get_method_from_name(video):
    methods = ["youtube", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "Celeb-real", "Celeb-synthesis", "YouTube-real"]
    for method in methods:
        if method in video:
            return method

def get_method(video, data_path):
    methods = os.listdir(os.path.join(data_path, "manipulated_sequences"))
    methods.extend(os.listdir(os.path.join(data_path, "original_sequences")))
    methods.append("Original")
    selected_method = ""
    for method in methods:
        if method in video:
            selected_method = method
            break
    return selected_method

def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs


def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k[:-4])
            else:
                originals.append(k[:-4])

    return originals, fakes

