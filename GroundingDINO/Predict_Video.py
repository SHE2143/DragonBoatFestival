from groundingdino.util.inference import load_model, predict, annotate
from transformers import AutoTokenizer, AutoModel
import cv2
import os

# 缓存 BERT 模型
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

# 提示词与阈值
TEXT_PROMPT = " people . bicycle . car. van . truck . tricycle . awning-tricycle . bus . motor"
BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5

# 类别映射
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

# 输入视频路径
VIDEO_PATH = "/Path/to/video"

# 输出目录
OUTPUT_IMAGE_DIR = "/root/GroundingDINO/output_video/images"
OUTPUT_LABEL_DIR = "/root/GroundingDINO/output_video/labels"
OUTPUT_VIDEO_PATH = "/root/GroundingDINO/output_video/detection_result.mp4"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0

# 获取视频参数（用于输出）
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改为 'XVID'、'avc1' 等
out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束

    print(f"Processing frame {frame_index}...")

    h, w = frame.shape[:2]
    image_source = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推理
    boxes, logits, phrases = predict(
        model=model,
        image=image_rgb,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # 保存图像帧
    image_name = f"frame_{frame_index:06d}.jpg"
    out_image_path = os.path.join(OUTPUT_IMAGE_DIR, image_name)
    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(out_image_path, annotated)

    # 写入视频帧
    out_video.write(annotated)

    # 保存标签
    out_label_path = os.path.join(OUTPUT_LABEL_DIR, f"frame_{frame_index:06d}.txt")
    with open(out_label_path, "w") as f:
        for box, phrase, logit in zip(boxes, phrases, logits):
            phrase = phrase.lower()
            class_id = PHRASE_TO_CLASSID.get(phrase, -1)
            if class_id == -1:
                continue

            x1, y1, x2, y2 = box.tolist()
            conf = float(logit)
            f.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")

    frame_index += 1

cap.release()
out_video.release()
print("✅ 视频处理完毕，检测结果保存在：")
print(f"📁 图像帧：{OUTPUT_IMAGE_DIR}")
print(f"📁 标签：{OUTPUT_LABEL_DIR}")
print(f"🎥 合成视频：{OUTPUT_VIDEO_PATH}")
