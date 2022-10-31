import argparse
# from ml.ml_camera import get_single_img, load_ml_model

from utilslocal.my_utils import camera_loop, get_highest_img_save_index


import torch

# from ml.ml_camera import PersonCheckerV2
# from utils import TryExcept
from itertools import compress
from PIL import Image
import cv2
import numpy as np
import sys
from functools  import partial
# sys.path.insert(0, 'c:\\Users\\garla\\git\\securitycamera\\src')
import time
# from playsound import playsound
import os
from gtts import gTTS


from pygame import mixer  # Load the popular external library
class PlaySound:
    # def __int__(self):
    #     
    
    def play(self):
        filename = r"C:\Users\garla\git\securitycamera\src\utilslocal\audio\scarysounds.mp3"
        # playsound(filename, block=False)
        mixer.init()
        mixer.music.load(filename)
        mixer.music.play()
        time.sleep(5)
        mixer.music.stop()




class PersonCheckerV2:
    # https://github.com/ultralytics/yolov5
    def __init__(self)  :
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
        
        
    def check_if_person_in_image(self,image,threshold=0.6):
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
    
    def show_bboxes_np(self, img_np, xyxy):
        for row in xyxy:
            p1 =int( row[0])
            p2 =int( row[1])
            p3 =int( row[2])
            p4 =int( row[3])
            
            pp1 = (p1,p2)
            pp2 = (p3,p4)
            lw = 5
            cv2.rectangle(img_np, pp1, pp2, (200,200,200), thickness=lw, lineType=cv2.LINE_AA)
        return img_np
        
    def show_bboxes(self, img, xyxy):
        img_np = np.array(img)
        img_np2 = self.show_bboxes_np(img_np, xyxy)        
        img_pil = Image.fromarray(img_np2)
        return img_pil, img_np
    
    
def parse_inputs():
    parser  = argparse.ArgumentParser('security camera capture and ML')
    parser.add_argument("--usbcam", action = 'store_true', default=False)
    parser.add_argument("--doml", action = 'store_true', default=False)
    parser.add_argument("--fakecamera", action = 'store_true', default=False)
    
    
    args = parser.parse_args()
    return args


def say_num_people(person_checker, img):
    numpeople,_ = person_checker.check_if_person_in_image(img)
    print(f"num people is {numpeople}")
    
def modify_frame(person_checker, img_np):
    img_np = img_np[180:,:,:] # crop out the top
    
    numpeople,xyxy = person_checker.check_if_person_in_image(img_np)
    print(f"num people is {numpeople}")
        
    frame = person_checker.show_bboxes_np(img_np,xyxy)
    if numpeople>0:
        sounds.play()
        if numpeople ==1:
            s = f"I see just one person. So you can only have one piece of candy. "
        else:
            s = f"I see {numpeople} people. You all can have {numpeople} pieces of candy. "
        file = "./file.mp3"
        if os.path.exists(file):
            os.unlink(file)
        tts = gTTS(s)
        tts.save(file)
        # from pygame import mixer
        mixer.init()
        mixer.music.load(file)
        mixer.music.play()
    return frame

if __name__ == "__main__":
    args = parse_inputs()
    print(args)
    if args.usbcam and args.doml==False:
        camera_loop(fake_camera=args.fakecamera)
        exit()
    if args.usbcam and args.doml==True:
        person_checker = PersonCheckerV2()
        sounds = PlaySound()
        # say_num_people2 = partial(say_num_people)
        modify_frame2 = partial(modify_frame,person_checker)
        camera_loop(None, modify_frame2, fake_camera=args.fakecamera)
        exit()
        
    
   
    # basically using this for testing. 
    # get_highest_img_save_index()
    # get_single_img()
    # load_ml_model()
        

