import cv2
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
import numpy as np
import os
import cv2
import torch
from statistics import mean

def transform_frame(image, image_size):
    transform_pipeline = Compose([
                IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REPLICATE)
                ]
            )
    return transform_pipeline(image=image)['image']
    
    
def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def custom_round(values, threshold):
    result = []
    for value in values:
        if value > threshold:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)

def custom_video_round(preds, threshold):
    for pred_value in preds:
        if pred_value > threshold:
            return pred_value
    return mean(preds)


def shuffle_dataset(dataset):
  import random
  # import numpy as np
  random.seed(4)
  random.shuffle(dataset)
  return dataset

    
def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    # print(preds)
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]
    # print(preds)
    correct = 0
    correct_positive = 0
    correct_negative = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
            if labels[i] == 0:
                correct_negative += 1
            elif labels[i] == 1:
                correct_positive += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class, correct_negative, correct_positive
