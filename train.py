import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from tqdm import tqdm
import yaml
import argparse
import time
import pickle
from torch.optim import lr_scheduler
from utils import  check_correct, resize, shuffle_dataset
from EffiViM.EViM import EffiViM
from multiprocessing.pool import Pool
from multiprocessing import Manager
from functools import partial
from deepfakes_dataset import DeepFakesDataset
import collections
import math

BASE_DIR = "./"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
MODELS_PATH = "models/model"



def read_frames(video_path, train_dataset, validation_dataset):
    # Get the video label based on dataset selected
    label = 0. if ("Original" in video_path) or ("real" in video_path) else 1.
    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_real']), 1)
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_fake']), 1)
    # if VALIDATION_DIR in video_path:
    #     min_video_frames = int(max(min_video_frames/5, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}
    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    # print(video_path, label)
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceSwap|NeuralTextures|ALL-FFPP|CDF)")
    parser.add_argument('--patience', type=int, default=50,
                        help="How many epochs wait before stopping for validation loss not improving.")

    # -------------Initialize the model and set hyperparameters-------------
    opt = parser.parse_args()
    print(opt)
    start_time = time.time()
    with open("config.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    channels = config["model"]["efficient_net_channels"]
    model = EffiViM()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    if os.path.exists(opt.resume):
        print("loaded:", opt.resume)
        model.load_state_dict(torch.load(opt.resume))
    else:
        print("No checkpoint loaded.")

    # -------------Read Dataset-------------
    if opt.dataset == "CDF":
        folders = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    elif opt.dataset == "ALL-FFPP":
        folders = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    else:
        folders = ["Original", opt.dataset]
    sets = [TRAINING_DIR, VALIDATION_DIR]
    print(f"Dataset Folders: {folders}")
    video_paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    video_paths.append(os.path.join(subfolder, video_folder_name))
    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(video_paths), desc="Load Dataset:") as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset), video_paths):
                pbar.update()

    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    print("___________________")
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING DATA__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(f"Real(0): {train_counters[0]}, Fake(1): {train_counters[1]}, Weights: {train_counters[0] / train_counters[1]}")

    print("__VALIDATION DATA__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(f"Real(0): {val_counters[0]}, Fake(1): {val_counters[1]}")
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_counters[0] / train_counters[1]]))
    train_labels = np.asarray([row[1] for row in train_dataset])
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    # Create DataLoader
    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset], dtype=object), train_labels, config['model']['image-size'])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset], dtype=object), validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset

    # -------------Start Train-------------
    model = model.cuda()
    train_counter = 0
    not_improved_loss_count = 0
    previous_loss = math.inf

    # 初始化一个字典来存储每轮训练和验证的损失值和准确率
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_acc = 0
    # Training and Validation
    for e in range(opt.num_epochs + 1):
        epoch_start_time = time.time()
        if not_improved_loss_count == opt.patience:
            break
        train_counter = 0
        total_train_loss = 0
        print(f"Epoch:{e + 1}/{opt.num_epochs}")
        # Training
        model.train()
        with tqdm(total=len(train_dl), desc=f'EPOCH #{e + 1} - Training', unit='batch', colour="red") as train_bar:
            train_correct = 0
            train_positive = 0
            train_negative = 0
            for index, (images, train_labels) in enumerate(train_dl):
                images = np.transpose(images, (0, 3, 1, 2))
                train_labels = train_labels.unsqueeze(1)
                images = images.cuda()
                y_pred = model(images)
                y_pred = y_pred.cpu()
                # print(y_pred, train_labels)
                loss = loss_fn(y_pred, train_labels)
                corrects, positive_class, negative_class, correct_negative, correct_positive = check_correct(y_pred, train_labels)
                train_correct += corrects
                train_positive += positive_class
                train_negative += negative_class
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_counter += 1
                total_train_loss += round(loss.item(), 2)
                train_bar.update(1)
            train_loss = total_train_loss / train_counter
            train_accuracy = train_correct / (train_counter * config['training']['bs'])
            print("\nTrain Loss: ", train_loss, "Train Accuracy: ", train_accuracy, "Train 0s: ", train_negative, "Train 1s:",train_positive)
        scheduler.step()
        train_correct /= train_samples
        total_train_loss /= train_counter

        # Validation
        val_counter = 0
        total_val_loss = 0
        model.eval()
        with tqdm(total=len(val_dl), desc=f'EPOCH #{e + 1} - Validation', unit='batch',colour="blue") as val_bar, torch.no_grad():
            val_correct_positive = 0
            val_correct_negetive = 0
            val_correct = 0
            val_positive = 0
            val_negative = 0
            for index, (val_images, val_labels) in enumerate(val_dl):
                val_images = np.transpose(val_images, (0, 3, 1, 2))
                val_images = val_images.cuda()
                val_labels = val_labels.unsqueeze(1)
                val_pred = model(val_images)
                val_pred = val_pred.cpu()
                # print(val_pred, val_labels)
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class, correct_negative, correct_positive = check_correct(val_pred, val_labels)
                # exit()
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_correct_positive += correct_positive
                val_correct_negetive += correct_negative
                val_counter += 1
                val_bar.update(1)
        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss_count += 1
        else:
            not_improved_loss_count = 0
        previous_loss = total_val_loss

        print("#" +
              str(e + 1) + "/" + str(opt.num_epochs) +
              "val_loss:" + str(total_val_loss) +
              "val_accuracy:" + str(val_correct) +
              "val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) +
              "val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)) +
              "acc0:" + str(val_correct_negetive) + "/" + str(np.count_nonzero(validation_labels == 0)) +
              "acc1:" + str(val_correct_positive) + "/" + str(np.count_nonzero(validation_labels == 1))
              )

        if val_correct > best_acc:
            best_acc = val_correct
            print(f"Save Best Results {best_acc}")
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            torch.save(model.state_dict(), os.path.join(MODELS_PATH, opt.dataset))

        # 将当前 epoch 的损失值和准确率添加到 history 字典中
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(total_val_loss)
        history['val_accuracy'].append(val_correct)

        file_path = 'models/train_history/training_history.pkl'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 将 history 字典保存到 pickle 文件中
        with open('models/train_history/training_history.pkl', 'wb') as f:
            pickle.dump(history, f)

    print(f"training time:{time.time() - start_time}")
        
        
