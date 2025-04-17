import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from PIL import Image
import numpy as np

# ==== 模型初始化 ====
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/Path/to/weight",   # 替换为你的模型路径
    confidence_threshold=0.25,
    device="0",  # "cpu" 或 "0"
)

# ==== 输入输出路径配置 ====
image_dir = "/Path/to/imgdir"  # 替换为图片路径
output_dir = "output_sahi_video"
output_image_dir = os.path.join(output_dir, "images")
output_label_dir = os.path.join(output_dir, "labels")
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ==== 滑窗参数 ====
SLICE_HEIGHT = 256
SLICE_WIDTH = 256
OVERLAP_RATIO = 0.2

# ==== 遍历图片 ====
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # ==== SAHI滑窗推理 ====
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
    )

    # ==== 生成YOLO标签 ====
    yolo_lines = []
    for det in result.object_prediction_list:
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        conf = det.score.value
        class_id = det.category.id  # 如果你训练是从0开始，这里去掉+1

        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}")

    # ==== 保存txt ====
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(output_label_dir, txt_name)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # ==== 保存可视化图像 ====
    vis_img = visualize_object_predictions(
        image=np.array(image),
        object_prediction_list=result.object_prediction_list,
    )
    out_img_path = os.path.join(output_image_dir, img_name)
    Image.fromarray(vis_img["image"]).save(out_img_path)

print("✅ SAHI 滑窗检测完成！输出结果已保存：")
print(f"📁 标签路径：{output_label_dir}")
print(f"📁 图像路径：{output_image_dir}")
