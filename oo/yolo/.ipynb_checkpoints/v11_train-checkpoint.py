import os
import torch
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from configs11 import get_configs
import warnings
warnings.filterwarnings('ignore')

def main():
    if not get_configs.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    model = YOLO(get_configs.model_yaml).load(get_configs.yolo_weight)
    model.train(**get_configs.train_args)


if __name__ == '__main__':
    main()
