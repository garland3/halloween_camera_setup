# %%

from pathlib import Path
from tkinter import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from ml.ml_camera import load_ml_model, get_single_img
from ml.ml_camera import model_meta_data
from ml.ml_camera import draw_prediction, draw_prediction_v2
from utils.my_utils import get_img_dir
import cv2
from PIL import Image
# from utils.my_utils import get_single_img
# from process_webcam_ml.

# %%
imgdir = get_img_dir()
print(imgdir.absolute())
# %%
image = get_single_img()
# %%
feature_extractor, model = load_ml_model()


# %%




inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

    img_size = torch.tensor([tuple(reversed(image.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    
    
# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)
# %%
# outputs

scores = processed_outputs[0]['scores'].numpy()
scores.shape

bboxes = processed_outputs[0]['boxes'].numpy()
bboxes.shape

label_id = processed_outputs[0]['labels'].numpy()

plt.plot(scores)


# %%

logits = outputs.logits
bboxes_raw = outputs.pred_boxes.squeeze()

# %%
metadata = model_meta_data()

# %%
data
# # %%
# logits2 = logits.squeeze()
# bboxes2 = bboxes.squeeze()
# # %%
# bboxes2.shape, logits2.shape
# %%
id2label=data['id2label']
id2label
# %%

# Width = img_size[1]
# Height = img_size[0]

Width = image.size[1]
Height = image.size[0]

boxes_for_overlay =[]
for score, bboxes_row,bbox_raw_row, label_id_row in zip(scores,bboxes, bboxes_raw,label_id):
    # class_id = np.argmax(logitsrow).item()
    # score = logitsrow[class_id]
    if score<0.3: continue
    class_id_str = str(label_id_row)
    if class_id_str in id2label.keys():
        label = id2label[class_id_str]
        print(f"{label} with score {score}, bb {bboxes_row} and bbox_raw_row {bbox_raw_row}")
        # center_x = int(bbox_raw_row[0] * Width)
        # center_y = int(bbox_raw_row[1] * Height)
        # w = int(bbox_raw_row[2] * Width)
        # h = int(bbox_raw_row[3] * Height)
        # x = center_x - w / 2
        # y = center_y - h / 2
        
        x = bboxes_row[0]
        y = bboxes_row[1]
        
        w = x-bboxes_row[2]
        h = y-bboxes_row[3]
        
        boxes_for_overlay.append([label, x, y, w, h])

# class_id

image_np = np.array(image)
# mycolor =  np.array([100,100,100],dtype=np.uint)
mycolor = (255,100,100)
for box in boxes_for_overlay:
    label = box[0]
    x =int( round( box[1]))
    y = int( round(box[2]))
    x_plus_w = int( round(box[3])+x)
    y_plus_h = int( round(box[4])+h)
    print( label,mycolor ,x,y,w,h)
    draw_prediction_v2(image_np, label,mycolor ,x,y,w,h)
    # cv2.rectangle(image_np, (x,y), (x_plus_w,y_plus_h), mycolor,5)
    
    

imagewithbox = Image.fromarray( image_np)

imagewithbox
# %%
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]
import io

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def visualize_prediction(pil_img, output_dict, threshold=0.7, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()
    if id2label is not None:
        labels = [id2label[x] for x in labels]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    return fig2img(plt.gcf())
# %%
inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

    img_size = torch.tensor([tuple(reversed(image.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    
# %%

# %%
metadata['id2label'] = { int(k):v for  k,v  in  metadata['id2label'].items()}

# %%

result = visualize_prediction(image, processed_outputs[0], 0.2, metadata['id2label'])
# %%
processed_outputs

# %%
