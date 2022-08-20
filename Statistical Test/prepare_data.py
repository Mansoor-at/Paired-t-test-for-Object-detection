import json
import os
import torch
import torchvision.ops.boxes as bops
import numpy as np
from compute_ap import compute_ap

# This code requires two files in json format.
# 1. Ground truth image names and boxes
# 2. model predictions json file

# To prepare a list containing image file names. We do this because the ground truth and prediction json files
# are do not contain images and their predictions in the same order
images = []
with open('gt_filenames.json', 'r') as data_file:
    gt_data = json.loads(data_file.read())

for value in gt_data['images']:
        name = value['file_name']
        filename, file_extension = os.path.splitext(name)
        images.append(filename)

# Load prediction json file and prepare a list with the format [box, class_id] = [x1, y1, width, height, class_id]
pred_boxes = []
with open('n1.json', 'r') as data_file:
    data = json.loads(data_file.read())

for c in range(len(images)):
    a = list(filter(lambda x:x["image_id"]==images[c], data))
    box1 = []
    for b in range(len(a)):
        image_id = a[b]["image_id"]
        cat_id = a[b]['category_id']
        bbox = a[b]["bbox"]
        score = a[b]["score"]
        if score > 0.30:
            bbox.append(cat_id)
            box1.append(bbox)
    pred_boxes.append(box1)

# Load ground truth json file and prepare a list with the format [box, class_id] = [x1, y1, width, height, class_id]

with open('g_truth.json', 'r') as data_file:
    g_data = json.loads(data_file.read())
gt_d = []
for c in range(len(images)):
    a = list(filter(lambda x:x["image_id"]==images[c], g_data))
    boxes = []
    for b in range(len(a)):
        # image_id = a[b]["image_id"]
        cat_id = a[b]['category_id']
        bbox = a[b]["bbox"]
        bbox.append(cat_id)
        boxes.append(bbox)
    gt_d.append(boxes)

# Here we compute IoU at 50% threshold
ious = []
AP = []
for v in range(260):
    c = len(gt_d[v])
    d = len(pred_boxes[v])
    corrects = []
    for j in range(len(gt_d[v])):
        a = gt_d[v][j]
        c1= a[4]
        a = a[0:4:1]
        # convert coco format to voc format
        a[2],  a[3] = a[0]+a[2], a[1]+a[3]
        for l in range(len(pred_boxes[v])):
            b = pred_boxes[v][l]
            c2 = b[4]
            b = b[0:4:1]
            # convert coco format to voc format
            b[2], b[3] = b[0] + b[2], b[1] + b[3]
            box1 = torch.tensor([a], dtype=torch.float)
            box2 = torch.tensor([b], dtype=torch.float)
            # Compute IoU
            iou = bops.box_iou(box1, box2)
            # We consider predictions as correct when both boxes have IoU > 50% and class id is matched.
            if c1 == c2 and iou > 0.50:
                corrects.append(1)
                ious.append(iou)
            else:
                corrects.append(0)
                ious.append(iou)

    true_positives = np.array(corrects)
    false_positives = 1 - true_positives
    # Following two lines were used in another implementation, but are not required here.
    # false_positives = np.cumsum(false_positives)
    # true_positives = np.cumsum(true_positives)

    # compute precision, recall and average precision
    recall = true_positives / len(gt_d[v]) if len(gt_d[v]) else true_positives
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    ap = compute_ap(recall, precision)
    AP.append(ap)

Ap = np.asarray(AP)
# saving the AP numpy array
np.save("n1_ap.npy", Ap)

