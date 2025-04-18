import json
import os
from PIL import Image


# 示例用法

coco_json_path = r'/root/dataset/name/annotations/xx.json'  # json 格式的 COCO 标注数据路径
images_dir = r'/root/dataset/name/images/'  # 图像文件夹路径
output_dir = r'/root/dataset/dataset/labels/'  # 输出 YOLO 格式的标注数据路径

os.makedirs(output_dir, exist_ok=True)
print(output_dir)


def convert_coco_yolo(coco_json_path, images_dir, output_dir):
    # 读取 COCO 格式的 JSON 文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建图片ID到文件名和尺寸的映射
    images_info = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}

    # 对每个标注进行处理
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id'] - 1  # 假设 COCO 类别从 1 开始，YOLO 从 0 开始
        bbox = ann['bbox']
        
        # 获取图片尺寸
        img_file, img_width, img_height = images_info[image_id]

        # 计算 YOLO 格式的中心点和宽高
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        # 构建 YOLO 格式的标注字符串
        yolo_format = f"{category_id} {x_center} {y_center} {width} {height}\n"

        # 确定输出文件名并写入数据
        output_filename = os.path.join(output_dir, f"{image_id}.txt")
        with open(output_filename, 'a') as file:
            file.write(yolo_format)



convert_coco_yolo(coco_json_path, images_dir, output_dir)
