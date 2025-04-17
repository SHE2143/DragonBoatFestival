import os
import numpy as np

def read_image_sizes(image_folder):
    image_sizes = {}
    for img_name in os.listdir(image_folder):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, img_name)
            from PIL import Image
            with Image.open(img_path) as img:
                width, height = img.size
            image_sizes[os.path.splitext(img_name)[0]] = (width, height)
    return image_sizes

def parse_txt_file(txt_path, is_gt=True):
    annotations = []
    if not os.path.exists(txt_path):
        return annotations
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_gt:
                class_id, xc, yc, w, h = map(float, parts)
                conf = 1.0
            else:
                class_id, xc, yc, w, h, conf = map(float, parts)
            img_name = os.path.splitext(os.path.basename(txt_path))[0]
            width, height = image_sizes[img_name]
            x1 = (xc - w/2) * width
            y1 = (yc - h/2) * height
            x2 = (xc + w/2) * width
            y2 = (yc + h/2) * height
            annotations.append([int(class_id), x1, y1, x2, y2, conf])
    return annotations

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            ap += (mrec[i] - mrec[i-1]) * mpre[i]
    return ap

def evaluate_detections(gt_folder, det_folder, image_folder, iou_threshold=0.5):
    global image_sizes
    image_sizes = read_image_sizes(image_folder)
    
    # 初始化数据存储
    gt_dict = {}  # {class_id: [[img_name, x1,y1,x2,y2], ...]}
    det_dict = {} # {class_id: [[img_name, x1,y1,x2,y2, conf], ...]}
    
    # 读取GT标注
    for gt_file in os.listdir(gt_folder):
        if not gt_file.endswith('.txt'):
            continue
        gt_path = os.path.join(gt_folder, gt_file)
        img_name = os.path.splitext(gt_file)[0]
        for ann in parse_txt_file(gt_path, is_gt=True):
            class_id = ann[0]
            if class_id not in gt_dict:
                gt_dict[class_id] = []
            gt_dict[class_id].append([img_name] + ann[1:5])
    
    # 读取检测结果
    for det_file in os.listdir(det_folder):
        if not det_file.endswith('.txt'):
            continue
        det_path = os.path.join(det_folder, det_file)
        img_name = os.path.splitext(det_file)[0]
        for ann in parse_txt_file(det_path, is_gt=False):
            class_id = ann[0]
            if class_id not in det_dict:
                det_dict[class_id] = []
            det_dict[class_id].append([img_name] + ann[1:6])
    
    # 初始化全局统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    aps = []
    
    # 遍历所有类别
    for class_id in gt_dict.keys():
        class_gt = gt_dict.get(class_id, [])
        class_det = det_dict.get(class_id, [])
        class_det.sort(key=lambda x: -x[5])  # 按置信度降序排列
        
        # 初始化匹配记录
        used_gt = {}
        tp = np.zeros(len(class_det))
        fp = np.zeros(len(class_det))
        
        # 遍历检测结果
        for det_idx, det in enumerate(class_det):
            img_name, x1, y1, x2, y2, conf = det
            best_iou = 0.0
            best_gt_idx = -1
            
            # 查找同图片同类的GT框
            for gt_idx, gt in enumerate(class_gt):
                if gt[0] != img_name:
                    continue
                gt_box = gt[1:]
                det_box = [x1, y1, x2, y2]
                iou = calculate_iou(gt_box, det_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 判断TP/FP
            if best_gt_idx != -1:
                if (img_name, best_gt_idx) not in used_gt:
                    tp[det_idx] = 1
                    used_gt[(img_name, best_gt_idx)] = True
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1
        
        # 统计当前类别的TP/FP/FN
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        class_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
        class_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
        class_fn = len(class_gt) - class_tp
        
        # 累加到全局统计
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        
        # 计算当前类别的AP
        recall = cum_tp / len(class_gt) if len(class_gt) > 0 else np.zeros_like(cum_tp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        ap = compute_ap(recall, precision)
        aps.append(ap)
        print(f'Class {class_id}: AP={ap:.4f}')
    
    # 计算全局指标
    mAP = np.mean(aps) if aps else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nGlobal Metrics:")
    print(f'mAP@IoU={iou_threshold}: {mAP:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    return mAP, precision, recall, f1

# 使用示例
if __name__ == '__main__':
    gt_folder = 'Path/to/your/gt/folder'
    det_folder = 'Path/to/your/detection/folder'
    image_folder = 'Origin/img/folder/or/Processed/img/folder'
    evaluate_detections(gt_folder, det_folder, image_folder, iou_threshold=0.5)