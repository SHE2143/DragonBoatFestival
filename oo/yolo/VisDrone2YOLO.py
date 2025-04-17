import os
import cv2
from glob import glob

# 类别映射（根据你训练需要选择哪些类）
# 例如只训练人 (category_id == 0)，可筛选只保留该类
# VisDrone官方类标号是从 0~10（或1~10）不一，建议手动检查确认
# 这里以 0 开始编号为例
valid_classes = list(range(10))  # 可根据需要裁剪类

# 路径
visdrone_ann_dir = '/root/annotations'
visdrone_img_dir = '/root/images'
output_img_dir = '/root/demo/img'
output_label_dir = '/root/demo/labels'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

ann_files = glob(os.path.join(visdrone_ann_dir, '*.txt'))

for ann_path in ann_files:
    filename = os.path.basename(ann_path).replace('.txt', '')
    img_path = os.path.join(visdrone_img_dir, filename + '.jpg')

    if not os.path.exists(img_path):
        continue

    # 读取图像尺寸
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 复制图像到yolo目录
    cv2.imwrite(os.path.join(output_img_dir, filename + '.jpg'), img)

    # 读取并转换标注
    with open(ann_path, 'r') as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue

        x1, y1, bw, bh = map(float, parts[:4])
        class_id = int(parts[5])

        if class_id not in valid_classes:
            continue

        # 转换为YOLO格式
        x_center = (x1 + bw / 2) / w
        y_center = (y1 + bh / 2) / h
        bw /= w
        bh /= h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
        yolo_lines.append(yolo_line)

    # 写入YOLO标签文件
    with open(os.path.join(output_label_dir, filename + '.txt'), 'w') as f:
        f.write('\n'.join(yolo_lines))
