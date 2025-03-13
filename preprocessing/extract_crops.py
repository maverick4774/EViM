import argparse
import json
import os
from os import cpu_count
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from multiprocessing.pool import Pool
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm
from utils import get_video_paths, get_method_from_name

def extract_video(video, root_dir, dataset):
    try:
        boxes_path = os.path.join(opt.data_path, "boxes", get_method_from_name(video), os.path.splitext(os.path.basename(video))[0] + ".json")
        if not os.path.exists(boxes_path) or not os.path.exists(video):
            return
        with open(boxes_path, "r") as box_f:
            boxes_dict = json.load(box_f)

        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success or str(i) not in boxes_dict:
                continue
            id = os.path.splitext(os.path.basename(video))[0]
            method = get_method_from_name(video)
            crops = []
            boxes = boxes_dict[str(i)]
            if boxes is None:
                continue
            else:
                counter += 1
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in box]
                w = xmax - xmin
                h = ymax - ymin
                p_h = 0
                p_w = 0
                if h > w:
                    p_w = int((h-w)/2)
                elif h < w:
                    p_h = int((w-h)/2)
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                h, w = crop.shape[:2]
                crops.append(crop)
            output_dir = os.path.join(opt.output_path, method)
            os.makedirs(os.path.join(output_dir, id), exist_ok=True)
            for j, crop in enumerate(crops):
                cv2.imwrite(os.path.join(output_dir, id, "{}_{}.png".format(i, j)), crop)
        if counter == 0:
            print(video, counter)
    except Exception as e:
        print("Error:", e)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="FFPP", type=str,
                        help='Dataset (FFPP / CDF')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='', type=str,
                        help='Output directory')

    opt = parser.parse_args()
    print(opt)
    dataset = 0 if opt.dataset.upper() == "CDF" else 1
    os.makedirs(opt.output_path, exist_ok=True)
    excluded_videos = os.listdir(opt.output_path)
    videos_paths = get_video_paths(opt.data_path, dataset)
    print("Video Numbers:", len(videos_paths), "Dataset:", opt.dataset)

    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(videos_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=opt.data_path, dataset=dataset), videos_paths):
                pbar.update()
