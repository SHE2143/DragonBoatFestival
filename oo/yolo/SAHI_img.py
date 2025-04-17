import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import save_json
from sahi.utils.cv import visualize_object_predictions
from PIL import Image
import numpy as np

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/Path/to/weight",
    confidence_threshold=0.25,
    device="0",
)

# 设置参数
image_dir = "/Path/to/imgdir"  # 替换为图片路径
output_txt_dir = "S/output_txt"
output_img_dir = "S/output_images"
os.makedirs(output_txt_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# 遍历图像
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path)
    w, h = image.size  # 获取原图尺寸

    # 获取预测结果（滑窗）
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # 生成YOLO格式内容
    yolo_lines = []
    for det in result.object_prediction_list:
        # 提取基础信息
        category_id = det.category.id
        confidence = det.score.value
        
        # 转换坐标为YOLO格式（归一化中心坐标）
        x1, y1, x2, y2 = det.bbox.to_xyxy()
        x_center = ((x1 + x2) / 2) / w  # 归一化中心x坐标
        y_center = ((y1 + y2) / 2) / h  # 归一化中心y坐标
        width = (x2 - x1) / w           # 归一化宽度
        height = (y2 - y1) / h          # 归一化高度

        # 按格式写入：class x_center y_center width height confidence
        line = f"{int(category_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}"
        yolo_lines.append(line)

    # 保存txt标签文件
    txt_filename = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(output_txt_dir, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # 保存可视化结果（可选）
    vis_img = visualize_object_predictions(
        image=np.array(image),
        object_prediction_list=result.object_prediction_list
    )
    out_img_path = os.path.join(output_img_dir, img_name)
    Image.fromarray(vis_img["image"]).save(out_img_path)