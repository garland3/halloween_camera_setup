import torch
from PIL import Image
# from utils.my_utils import get_imgs_in_dir


        
class PersonCheckerV2:
    # https://github.com/ultralytics/yolov5
    def __init__(self)  :
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
        
        
    def check_if_person_in_image(self,image,threshold=0.3):
        """
        Image can be an PIL image, path string, or path objet
        """
        results = self.model(image)
        xyxy = results.xyxy[0].detach().numpy()
        score = xyxy[:,4]
        label =[int(a) for a in xyxy[:,5]]
        mask = score>threshold
        label2 = label[mask]
        return label2.count()
    
# def get_single_img():
#     imgs = get_imgs_in_dir()
#     file = r"C:\Users\garla\git\securitycamera\imgs\img2223.png"
#     # file = imgs[0]
#     img = Image.open(file)
#     print(f"test img size is {img.size}")
#     return img


# from pathlib import Path
# import requests
# import json
# from operator import mod
# from transformers import YolosFeatureExtractor, YolosModel, YolosForObjectDetection
# import torch
# import cv2
# import mediapipe as mp
# import numpy as np
# from matplotlib import pyplot as plt
# import io

# Bounding Box
# MODEL= 'hustvl/yolos-small'


# from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

# def load_ml_model():
#     feature_extractor = YolosFeatureExtractor.from_pretrained(MODEL)
#     model = YolosForObjectDetection.from_pretrained(MODEL)
#     return feature_extractor, model

# def model_meta_data():
#     d = requests.get(f"https://huggingface.co/{MODEL}/raw/main/config.json")
#     folder = Path("mldata")
#     folder.mkdir(exist_ok=True)
#     with open(str(folder/ "config.json"), "w") as f:
#         data = d.json()
#         json.dump(data,f)
#         data['id2label'] = { int(k):v for  k,v  in  data['id2label'].items()}
#     return data

# def make_prediction(image, model, feature_extractor):
#     inputs = feature_extractor(image, return_tensors="pt")    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         img_size = torch.tensor([tuple(reversed(image.size))])
#         processed_outputs = feature_extractor.post_process(outputs, img_size)
# #     return processed_outputs
  
# class PersonChecker:
#     def __init__(self)  :
#         self.feature_extractor, self.model = load_ml_model()
        
#     def check_if_person_in_image(self,image,threshold=0.3):
#         """
#         Image can be an PIL image, path string, or path objet
#         """
#         if type(image) == Path or type(image) == str:
#             image = Image.open(image)
#         processed_outputs = make_prediction(image, self.model, self.feature_extractor)
#         output_dict = processed_outputs[0]
#         keep = output_dict["scores"] > threshold
#         labels = output_dict["labels"][keep].tolist()
#         return labels.count(1)
    

# COLORS = [
#     [0.000, 0.447, 0.741],
#     [0.850, 0.325, 0.098],
#     [0.929, 0.694, 0.125],
#     [0.494, 0.184, 0.556],
#     [0.466, 0.674, 0.188],
#     [0.301, 0.745, 0.933]
# ]
# import io

# def fig2img(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf)
#     buf.seek(0)
#     img = Image.open(buf)
#     return img

# def visualize_prediction(pil_img, output_dict, threshold=0.7, id2label=None):
#     keep = output_dict["scores"] > threshold
#     boxes = output_dict["boxes"][keep].tolist()
#     scores = output_dict["scores"][keep].tolist()
#     labels = output_dict["labels"][keep].tolist()
#     if id2label is not None:
#         labels = [id2label[x] for x in labels]

#     plt.figure(figsize=(16, 10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
#         ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
#     plt.axis("off")
#     return fig2img(plt.gcf())



# def find_person()
# inputs = feature_extractor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

#     img_size = torch.tensor([tuple(reversed(image.size))])
#     processed_outputs = feature_extractor.post_process(outputs, img_size)
    
    
# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, 
#                     classes, COLORS):
#     label = str(classes[class_id])
#     color = COLORS[class_id]
#     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
#     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
# def draw_prediction_v2(img, label, color, x, y, x_plus_w, y_plus_h):
#     # label = str(classes[class_id])
#     # color = COLORS[class_id]
#     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
#     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# inputs = feature_extractor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)
