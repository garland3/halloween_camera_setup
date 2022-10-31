# from flask import Flask, render_template, Response, make_response
from typing import List
import cv2
import base64
from pathlib import Path
import argparse
import re
import requests
import json

def get_img_dir()->Path:
    outputdir = Path("../imgs")
    outputdir.mkdir(exist_ok=True)
    return outputdir

def get_imgs_in_dir() -> List[Path]:
    dir = get_img_dir()
    imgs = list(dir.glob("*.png"))
    return imgs

def get_and_sort_imgs_by_number():
    imgs =get_imgs_in_dir()
    numbers  = [re.findall(r"\d+",str(a.name))[0] for a in imgs]
    # print()
    # for n,i in zip(numbers, imgs):
    #     print(n,i.name)
    numbers_ints = [int(a) for a in numbers]
    return numbers_ints

def get_highest_img_save_index()->int:
    numbers_ints = get_and_sort_imgs_by_number()
    if len(numbers_ints)>0:
        mymax =  max(numbers_ints)
    else:
        mymax = 0
    print(f"The max number is {mymax}")
    return mymax

class FakeCamera:
    def __init__(self):
        numbers_ints = get_and_sort_imgs_by_number()
        self.mymax =  max(numbers_ints)
        self.dir = get_img_dir()
        self.cnt = 1

    def read(self):
        name = self.dir / f"img{self.cnt}.png"
        print(f"Fake camera is using {name} as image ")
        img   = cv2.imread(name)
        self.cnt +=1
        if self.cnt>self.mymax:
            self.cnt = 1
        return None, img

    def release(self):
        print("Fake camera is done. ")

def camera_loop(fn_callback = None, fn_modify_frame = None, fake_camera = False):
    """
    fn_callback is called like this.  fn_callback(frame)
    frame is the frame from cv2
    
    ALSO
     frame = fn_modify_frame(frame)
    """
    outputdir = get_img_dir()
    print("starting camera")
    
    camera = cv2.VideoCapture(1) if fake_camera==False else FakeCamera()
    print("camera started")
    
    cnt = get_highest_img_save_index()
    print("Loop started. ")
    while True:
        ret, frame = camera.read()  
        # if fn_modify_frame_before_ml is not None:
        if fn_modify_frame is not None:
            frame = fn_modify_frame(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        cnt+=1
        if fake_camera==False:
            # Only save the files if it is from the real camera. 
            img_name = outputdir / f"img{cnt}.png"
            cv2.imwrite(str(img_name), frame)
        if fn_callback is not None:
            fn_callback(frame)
        
    camera.release()
    cv2.destroyAllWindows()

def parse_inputs():
    parser  = argparse.ArgumentParser('security camera capture and ML')
    parser.add_argument("--usbcam", action = 'store_true', default=False)
    args = parser.parse_args()
    return args


# if __name__ == "__main__":
#     args = parse_inputs()
#     print(args)
#     if args.usbcam:
#         camera_loop()
#     else:
#         get_highest_img_save_index()
        

