import os
import shutil
import random

# 原始目录
image_dir = "../datasets/DATA/images"  # DATA为datasets下的数据集名称
label_dir = "../datasets/DATA/labels"

# 输出目录
output_base = "../datasets/DATA"
splits = ["train", "val", "test"]
split_ratio = [0.8, 0.1, 0.1]  # 8:1:1 分割

# 获取所有图像文件（确保扩展名正确）
all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# 计算划分数量
total = len(all_images)
train_end = int(split_ratio[0] * total)
val_end = train_end + int(split_ratio[1] * total)

split_data = {
    "train": all_images[:train_end],
    "val": all_images[train_end:val_end],
    "test": all_images[val_end:]
}

# 创建输出目录结构
for split in splits:
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)

# 拷贝文件
for split, filenames in split_data.items():
    for img_file in filenames:
        # 拷贝图像
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(output_base, split, "images", img_file)
        shutil.copy(src_img, dst_img)

        # 拷贝对应标签（.txt）
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_lbl = os.path.join(label_dir, label_file)
        dst_lbl = os.path.join(output_base, split, "labels", label_file)
        if os.path.exists(src_lbl):  # 防止漏标
            shutil.copy(src_lbl, dst_lbl)
