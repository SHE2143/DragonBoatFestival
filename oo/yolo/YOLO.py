import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 初始化YOLO模型
model_path = "/Path/to/weight"
detection_model = YOLO(model_path)  # 加载训练好的YOLO模型

# 设置参数
image_dir = "/Path/to/imgdir"
output_txt_dir = "Y/output_txt"
output_img_dir = "Y/output_images"
os.makedirs(output_txt_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# 遍历图像
for img_name in os.listdir(image_dir):
    if not img_name.endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path)

    # 进行推理（推理会自动返回图片及预测结果）
    results = detection_model(img_path)

    # 获取推理结果
    predictions = results[0].boxes  # 获取预测的边界框
    class_ids = predictions.cls.cpu().numpy().astype(int)
    xyxy = predictions.xyxy.cpu().numpy()  # 预测框的坐标
    scores = predictions.conf.cpu().numpy()  # 置信度分数

    # 保存txt标签
    w, h = image.size
    yolo_lines = []
    for i in range(len(class_ids)):
        x1, y1, x2, y2 = xyxy[i]
        category_id = class_ids[i]
        confidence = scores[i]

        # 转换为YOLO格式（归一化中心坐标和宽高）
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_width = (x2 - x1) / w
        bbox_height = (y2 - y1) / h

        # 根据阈值筛选预测框
        if confidence >= 0.25:  # 使用阈值过滤低置信度的框
            yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {confidence:.6f}")

    # 保存txt文件
    txt_path = os.path.join(output_txt_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # 可视化预测框并保存图片
    vis_img = np.array(image)
    for i in range(len(class_ids)):
        if scores[i] >= 0.25:  # 可视化仅显示置信度大于阈值的框
            x1, y1, x2, y2 = xyxy[i]
            vis_img = cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    out_img_path = os.path.join(output_img_dir, img_name)
    Image.fromarray(vis_img).save(out_img_path)
