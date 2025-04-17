import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==== 模型路径与初始化 ====
model_path = "/Path/to/weight"
detection_model = YOLO(model_path)

# ==== 输入输出路径设置 ====
image_dir = "/Path/to/imgdir"
output_dir = "output_yolo_video"
output_image_dir = os.path.join(output_dir, "images")
output_label_dir = os.path.join(output_dir, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ==== 推理参数 ====
CONFIDENCE_THRESHOLD = 0.25  # 置信度阈值

# ==== 遍历图像文件 ====
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # ==== 模型推理 ====
    results = detection_model(img_path)
    predictions = results[0].boxes
    class_ids = predictions.cls.cpu().numpy().astype(int)
    xyxy = predictions.xyxy.cpu().numpy()
    scores = predictions.conf.cpu().numpy()

    # ==== 保存标签（YOLO格式） ====
    yolo_lines = []
    for i in range(len(class_ids)):
        conf = scores[i]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = xyxy[i]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_width = (x2 - x1) / w
        bbox_height = (y2 - y1) / h
        class_id = class_ids[i]  # 如果训练时是从0开始可以去掉 +1

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {conf:.6f}")

    txt_name = os.path.splitext(img_name)[0] + ".txt"
    with open(os.path.join(output_label_dir, txt_name), "w") as f:
        f.write("\n".join(yolo_lines))

    # ==== 可视化并保存图片 ====
    vis_img = np.array(image)
    for i in range(len(class_ids)):
        conf = scores[i]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = xyxy[i]
        class_id = class_ids[i]
        label = f"ID:{class_id} {conf:.2f}"
        vis_img = cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        vis_img = cv2.putText(vis_img, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_img_path = os.path.join(output_image_dir, img_name)
    Image.fromarray(vis_img).save(out_img_path)

print("✅ 推理完成！输出路径如下：")
print(f"📁 标签目录：{output_label_dir}")
print(f"📁 图片目录：{output_image_dir}")
