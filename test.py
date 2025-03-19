import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from EffiViM.EViM import EffiViM
from multiprocessing.pool import Pool
from multiprocessing import Manager
from tqdm import tqdm
import yaml
import argparse
from sklearn.metrics import auc, accuracy_score, f1_score, roc_curve
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
from utils import custom_round, custom_video_round


BASE_DIR = "./"
RESULTS_DIR = "models"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "test_results")

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])
def AUC(correct_labels, preds):
    fpr, tpr, th = roc_curve(correct_labels, preds)
    model_auc = auc(fpr, tpr)
    return model_auc
    # print("AUC", model_auc)
def read_frames(video_path, videos):
    # Get the video label based on dataset selected
    if "Original" in video_path or "real" in video_path: # or "test" in video_path
        label = 0.
    else:
        label = 1.
    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}
    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0, 3):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # print(frames_paths_dict)
    # Select only the frames at a certain interval
    if frames_interval > 0:
        # print(f"帧间隔:{frames_interval}")
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict[key]) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]
    # print(f"{video_path}dict:---", frames_paths_dict, len(frames_paths_dict[0]))
    # exit()
    # Select N frames from the collected ones
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            # image = np.asarray(resize(cv2.imread(os.path.join(video_path, frame_image)), IMAGE_SIZE))
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    # video保存了从这个视频中提取的每个人脸的帧信息 label表示这个视频的标签 video_path是视频路径
    videos.append((video, label, video_path))


# Main body
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC',
                        help="Which dataset to use (Deepfakes|Face2Face|FaceSwap|NeuralTextures|ALL-FFPP|CDF)")
    parser.add_argument('--frames_per_video', type=int, default=30,
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.35, help="threshold")

    opt = parser.parse_args()
    print(opt)

    with open("config.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    channels = config["model"]["efficient_net_channels"]
    if os.path.exists(opt.checkpoint):
        print(f"Load model:{opt.checkpoint}")
        model = EffiViM()
        model.load_state_dict(torch.load(opt.checkpoint), strict=False)
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.checkpoint)
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, opt.dataset)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    preds = []


    if opt.dataset == "CDF":
        folders = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    elif opt.dataset == "ALL-FFPP":
        folders = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    else:
        folders = ["Original", opt.dataset]

    video_paths = []
    for folder in folders:
        subfolder = os.path.join(TEST_DIR, folder)
        for video in os.listdir(subfolder):
            video_paths.append(os.path.join(subfolder, video))


    mgr = Manager()
    videos = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(video_paths), desc="Load Dataset") as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos), video_paths):
                pbar.update()



    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []  # 所有视频的预测结果列表
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    f = open(os.path.join(OUTPUT_DIR, opt.dataset + model_name + ".txt"), "w+")
    with tqdm(total=len(videos), desc="Predicting", colour="blue") as pbar, torch.no_grad():
        for index, video in enumerate(videos):  # videos是所有的视频， 这个for循环循环视频数量
            video_faces_preds = []  # 单个视频的预测结果列表
            video_name = video_names[index]
            f.write(os.path.basename(video_name))
            frames_count = sum(len(value) for value in video.values())  # 统计当前视频用了多少帧
            # print("帧数：", frames_count)
            for key in video:  # video是一个字典 key的值代表了人脸的编号， 这个for循环循环在这个视频下面的人脸编号
                faces_preds = []  # 所有人脸的所有帧的预测结果列表
                video_faces = video[key]  # video_faces代表了这个人脸key的所有人脸
                for i in range(0, len(video_faces), opt.batch_size):  # 这个for循环循环在这个人脸下的所有帧
                    faces = video_faces[i:i + opt.batch_size]  # 一个batch的人脸
                    faces = torch.tensor(np.asarray(faces))
                    if faces.shape[0] == 0:
                        continue
                    faces = np.transpose(faces, (0, 3, 1, 2))
                    faces = faces.cuda().float()
                    pred = model(faces)
                    scaled_pred = []
                    frame_count_weight = len(pred) / frames_count  # 计算这张脸在所有帧数中的权重
                    for idx, p in enumerate(pred):
                        scaled_pred.append(torch.sigmoid(p) * frame_count_weight)
                    faces_preds.extend(scaled_pred)
                    # print(faces_preds)
                current_faces_pred = sum(faces_preds) / len(faces_preds)
                face_pred = current_faces_pred.cpu().detach().numpy()[0]
                # print(frame_count_weight, face_pred)
                # print(face_pred)
                f.write(f" Face[{str(key)}]->{str(face_pred)} weight({len(pred)} / {frames_count})")
                video_faces_preds.append(face_pred)
            pbar.update(1)
            if len(video_faces_preds) > 1:
                video_pred = custom_video_round(video_faces_preds, opt.threshold)
            else:
                video_pred = video_faces_preds[0]
            preds.append([video_pred])
            rounded_pred = custom_round([video_pred], opt.threshold)
            if rounded_pred[0] == correct_test_labels[index]:
                result_text = " Correct prediction"
            else:
                result_text = " Wrong prediction"
            # print(video_name, video_faces_preds, correct_test_labels[index], result_text)
            f.write(" --> " + str(video_pred) + "(CORRECT LABEL: " + str(correct_test_labels[index]) + ")" + result_text +"\n")
    f.close()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)
    loss = loss_fn(tensor_preds, tensor_labels).numpy()
    accuracy = accuracy_score(custom_round(np.asarray(preds), opt.threshold), correct_test_labels)
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds), opt.threshold))
    auc = AUC(correct_test_labels, preds)
    print(f"{model_name} ACC:{accuracy}, Loss:{loss}, F1:{f1}, AUC:{auc}")



