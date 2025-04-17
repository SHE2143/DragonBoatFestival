from groundingdino.util.inference import load_model, load_image, predict, annotate
from transformers import AutoTokenizer, AutoModel
import cv2
import os

# 提前缓存 BERT 模型，避免推理时重复下载
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")

# 加载 GroundingDINO 模型
model = load_model(
    "/root/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/root/GroundingDINO/weights/groundingdino_swint_ogc.pth"
)


# 加载 GroundingDINO 模型
# model = load_model(
#     "/root/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
#     "/root/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
# )

# 固定提示词与阈值
TEXT_PROMPT = " people . bicycle . car. van . truck . tricycle . awning-tricycle . bus . motor"
BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5

# 类别映射（你可以扩展，改）
PHRASE_TO_CLASSID = {
    "pedestrain": 1,
    "people": 2,
    "bicycle": 3,
    "car": 4,
    "van": 5,
    "truck": 6,
    "tricycle": 7,
    "awning tricycle": 8,
    "bus": 9,
    "motor": 10,
}

# 输入与输出目录
INPUT_DIR = "/Path/to/imgdir"
OUTPUT_IMAGE_DIR = "/root/GroundingDINO/output/images"
OUTPUT_LABEL_DIR = "/root/GroundingDINO/output/labels"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 遍历目录中所有图片
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    print(f"Processing {filename}...")

    # 加载图像
    image_source, image = load_image(image_path)
    h, w = image_source.shape[:2]

    # 推理
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # 保存标注图像
    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    out_image_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
    cv2.imwrite(out_image_path, annotated)

    # 保存 YOLO 格式 label
    out_label_path = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
    with open(out_label_path, "w") as f:
        for box, phrase, logit in zip(boxes, phrases, logits):
            phrase = phrase.lower()
            class_id = PHRASE_TO_CLASSID.get(phrase, -1)
            if class_id == -1:
                continue  # 跳过未定义的类别

            x1, y1, x2, y2 = box.tolist()
            # x_center = (x1 + x2) / 2 / w
            # y_center = (y1 + y2) / 2 / h
            # box_width = abs(x2 - x1) / w
            # box_height = abs(y2 - y1) / h
            conf = float(logit)

            f.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

print("✅ 所有图片处理完毕，检测结果保存在 output/images 和 output/labels 文件夹中。")
