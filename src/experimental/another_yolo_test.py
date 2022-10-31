# %%

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# # %%
# results
# # %%
# vars(results).keys()
# # %%
# results.names
# # %%
# results.ims[0].shape
# # %%
# type(results)
# # %%
# df = results.pandas().xyxy[0]
# # %%
# df
# # %%
# results.xyxy
# # %%
# from PIL import Image
# # %%
# imgpil = Image.open(img)
# # %%
# results.xyxy[0][:,4]
# # %%
# results
# %%
import torch

# from ml.ml_camera import PersonCheckerV2
# from utils import TryExcept
from itertools import compress
from PIL import Image
import cv2
import numpy as np

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
        # return xyxy
        # print(xyxy)
        score = xyxy[:,4]
        label =[int(a) for a in xyxy[:,5]]
        mask = score>threshold
        # print(mask)
        # label2 = label[mask]
        label2 = list(compress(label,mask))
        
        return label2.count(0), xyxy
    
    def show_bboxes(self, img, xyxy):
        img_np = np.array(img)
        for row in xyxy:
            p1 =int( row[0])
            p2 =int( row[1])
            p3 =int( row[2])
            p4 =int( row[3])
            
            pp1 = (p1,p2)
            pp2 = (p3,p4)
            lw = 5
            cv2.rectangle(img_np, pp1, pp2, (200,200,200), thickness=lw, lineType=cv2.LINE_AA)
        img_pil = Image.fromarray(img_np)
        return img_pil
    

# %%
personchecker = PersonCheckerV2()
# image = get_single_img()

file = r"C:\Users\garla\git\securitycamera\imgs\img2223.png"
# file = r"C:\Users\garla\git\securitycamera\imgs\img10.png"
img = Image.open(file)

# xyxy = personchecker.check_if_person_in_image(file)
personcount,xyxy = personchecker.check_if_person_in_image(img)
personcount

newimg = personchecker.show_bboxes(img, xyxy)
newimg
# %%
