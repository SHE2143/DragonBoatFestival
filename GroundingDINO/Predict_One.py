from groundingdino.util.inference import load_model, load_image, predict, annotate
from transformers import AutoTokenizer, AutoModel
import cv2

# 提前缓存 BERT 模型，避免推理时重复下载
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")

# 加载 GroundingDINO 模型
# model = load_model(
#     "/root/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     "/root/GroundingDINO/weights/groundingdino_swint_ogc.pth"
# )

model = load_model(
    "/root/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "/root/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
)


# 图像路径与描述
IMAGE_PATH = "/root/GroundingDINO/demo/dataset/1.jpg"
TEXT_PROMPT = "person ."
BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.4

# 读取图像
image_source, image = load_image(IMAGE_PATH)

# 进行推理
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# 保存标注图像
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)

PHRASE_TO_CLASSID = {
    "person": 0,
    "car": 1,
    "dog": 2,
    # 加你需要的映射
}

# 获取图像尺寸（用于归一化）
h, w = image_source.shape[:2]

with open("detection_yolo.txt", "w") as f:
    for box, phrase, logit in zip(boxes, phrases, logits):
        phrase = phrase.lower()
        # 映射 class id
        class_id = PHRASE_TO_CLASSID.get(phrase, -1)  # 如果找不到，返回 -1
        if class_id == -1:
            continue  # 跳过未定义的类别

        x1, y1, x2, y2 = box.tolist()

        # 计算中心点和宽高（YOLO格式）
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        box_width = (x1 - x2) 
        box_height = (y1 - y2) 

        conf = float(logit)

        # 写入文件
        f.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")