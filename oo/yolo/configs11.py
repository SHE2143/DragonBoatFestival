import os
import torch

class get_configs:
    # 通用配置
    root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))  # 当前文件的目录路径
    yolo_weight = os.path.join(root_path, 'models', 'yolo11n.pt')
    model_yaml = os.path.join(root_path, 'models', 'yolo11.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_wandb = True  # 是否启用wandb

    data = os.path.join(os.path.dirname(root_path), 'datasets', 'VisDrone.yaml')
    imageSize = 640 # 图片大小
    project = os.path.join(root_path, 'results')  # 指定项目文件夹

    # 定义训练参数为字典
    train_args = {
        'data': data,
        'imgsz': imageSize,
        'project': project,  # 指定项目文件夹
        'name': 'detect_train',  # 指定本次运行的名称

        # 训练参数
        'lr0': 0.01, # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        'lrf': 0.01, # (float) final learning rate (lr0 * lrf)
        'workers': 8, 
        'batch': 32,
        'epochs': 100,
        'amp': True,
        'cache': True,
        
        # 数据增强参数
        'crop_fraction': 0.8,  # 分类图像裁剪分数，1表示没有
        'dropout': 0.1,  # 随机失活概率
        'multi_scale': True,  # 启用多尺度训练
        'mosaic': 0.2,  # 马赛克数据增强概率
        'mixup': 0.2,  # minup增强概率
        'hsv_h': 0.015, # (float) image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7, # (float) image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4, # (float) image HSV-Value augmentation (fraction)
        'bgr': 0.1, # (float) image channel BGR (probability)
        'scale': 0.8, # (float) image scale (+/- gain)
        'shear': 0.3, # (float) image shear (+/- deg)
        'degrees': 0.3, # (float) image rotation (+/- deg)

        # 'fraction': 0.1  # 训练集筛选比例
    }

print(get_configs().root_path)