# %%

from collections import Counter
from tkinter import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from ml.ml_camera import load_ml_model, get_single_img, make_prediction
from ml.ml_camera import model_meta_data
from ml.ml_camera import PersonChecker
from utils.my_utils import get_img_dir
import cv2
from PIL import Image

# %%
imgdir = get_img_dir()
print(imgdir.absolute())
# %%
image = get_single_img()
sz = image.size
image = image.crop((sz[0]//2,int(45*sz[1]//100),sz[0],sz[1]))

feature_extractor, model = load_ml_model()

processed_outputs = make_prediction(image, model, feature_extractor)
output_dict = processed_outputs[0]
threshold=0.3
keep = output_dict["scores"] > threshold
labels = output_dict["labels"][keep].tolist()
labels
if 1 in labels:
    print("found peson")
    # return labels.count(1)

    


# %%

# Counter(labels)

# inputs = feature_extractor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

#     img_size = torch.tensor([tuple(reversed(image.size))])
#     processed_outputs = feature_extractor.post_process(outputs, img_size)
    
# %%
# mygen = 
imgdir = get_img_dir()
pngs = list(imgdir.glob("*.png"))
pngs
# %%
# mygen = check_if_person_in_image_generator()
personchecker = PersonChecker()

# %%
for png in pngs:
    # print(type(png))
    value = personchecker.check_if_person_in_image(str(png))
    print(value, png)
# %%
